use super::*;
use crate::primitives::PageID;
use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;

pub struct PageAllocator {
    pages: Arc<Mutex<HashMap<PageID, Page>>>,
    default_location: PageLocation,
}

impl PageAllocator {
    pub fn new(default_location: PageLocation) -> Self {
        Self {
            pages: Arc::new(Mutex::new(HashMap::new())),
            default_location,
        }
    }

    pub fn from_config(config: PageAllocatorConfig) -> Self {
        Self {
            pages: Arc::new(Mutex::new(HashMap::with_capacity(config.initial_capacity))),
            default_location: config.default_location,
        }
    }

    pub fn with_page<F, R>(&self, id: PageID, f: F) -> Option<R>
    where
        F: FnOnce(&Page) -> R,
    {
        let pages = self.pages.lock();
        pages.get(&id).map(f)
    }

    pub fn with_page_mut<F, R>(&self, id: PageID, f: F) -> Option<R>
    where
        F: FnOnce(&mut Page) -> R,
    {
        let mut pages = self.pages.lock();
        pages.get_mut(&id).map(f)
    }

    pub fn snapshot(&self, id: PageID) -> Option<PageSnapshotData> {
        self.with_page(id, |page| PageSnapshotData {
            page_id: id,
            data: page.data_slice().to_vec(),
            location: page.location(),
            epoch: page.epoch(),
        })
    }

    pub fn restore(&self, snapshot: &PageSnapshotData) -> Result<(), PageError> {
        self.allocate(snapshot.page_id, snapshot.data.len())?;
        self.with_page_mut(snapshot.page_id, |page| {
            page.data_mut_slice().copy_from_slice(&snapshot.data);
            page.set_epoch(snapshot.epoch);
        });
        Ok(())
    }

    pub fn snapshot_pages(&self) -> Vec<PageSnapshotData> {
        let pages = self.pages.lock();
        pages
            .iter()
            .map(|(id, page): (&PageID, &Page)| PageSnapshotData {
                page_id: *id,
                data: page.data_slice().to_vec(),
                location: page.location(),
                epoch: page.epoch(),
            })
            .collect()
    }

    pub fn page_infos(&self) -> Vec<PageSnapshotData> {
        self.snapshot_pages()
    }
}

impl PageAllocatorLike for PageAllocator {
    fn allocate(&self, id: PageID, size: usize) -> Result<(), PageError> {
        let page = Page::new(id, size, self.default_location)?;
        self.pages.lock().insert(id, page);
        Ok(())
    }

    fn free(&self, id: PageID) {
        self.pages.lock().remove(&id);
    }

    fn get(&self, _id: PageID) -> Option<&Page> {
        // Cannot return reference due to Mutex guard lifetime
        // Use with_page or with_page_mut instead
        None
    }
}
