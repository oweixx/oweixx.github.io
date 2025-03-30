// get the ninja-keys element
const ninja = document.querySelector('ninja-keys');

// add the home and posts menu items
ninja.data = [{
    id: "nav-about",
    title: "about",
    section: "Navigation",
    handler: () => {
      window.location.href = "/";
    },
  },{id: "nav-blog",
          title: "blog",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/blog/";
          },
        },{id: "post-3월-월간-회고",
      
        title: "3월 월간 회고",
      
      description: "월간 회고",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/2025/monthly_review_2503/";
        
      },
    },{id: "post-랜덤-마라톤-코스-41-42",
      
        title: "랜덤 마라톤 코스(41,42)",
      
      description: "랜덤 마라톤 코스(41,42)",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/2025/marathon_42_41/";
        
      },
    },{id: "post-랜덤-마라톤-코스-43",
      
        title: "랜덤 마라톤 코스(43)",
      
      description: "랜덤 마라톤 코스(43)",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/2025/marathon_43/";
        
      },
    },{id: "post-뤼카의-정리",
      
        title: "뤼카의 정리",
      
      description: "뤼카의 정리",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/2025/Lucas_theorem/";
        
      },
    },{id: "post-boj-1210-마피아-python",
      
        title: "BOJ 1210 마피아 (Python)",
      
      description: "BOJ 1210 마피아 (Python)",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/2025/BOJ1210/";
        
      },
    },{id: "news-a-simple-inline-announcement",
          title: 'A simple inline announcement.',
          description: "",
          section: "News",},{id: "projects-project-1",
          title: 'project 1',
          description: "with background image",
          section: "Projects",handler: () => {
              window.location.href = "/projects/1_project/";
            },},{id: "projects-project-2",
          title: 'project 2',
          description: "a project with a background image and giscus comments",
          section: "Projects",handler: () => {
              window.location.href = "/projects/2_project/";
            },},{id: "projects-project-3-with-very-long-name",
          title: 'project 3 with very long name',
          description: "a project that redirects to another website",
          section: "Projects",handler: () => {
              window.location.href = "/projects/3_project/";
            },},{
        id: 'social-email',
        title: 'email',
        section: 'Socials',
        handler: () => {
          window.open("mailto:%79%6F%75@%65%78%61%6D%70%6C%65.%63%6F%6D", "_blank");
        },
      },{
        id: 'social-inspire',
        title: 'Inspire HEP',
        section: 'Socials',
        handler: () => {
          window.open("https://inspirehep.net/authors/1010907", "_blank");
        },
      },{
        id: 'social-rss',
        title: 'RSS Feed',
        section: 'Socials',
        handler: () => {
          window.open("/feed.xml", "_blank");
        },
      },{
        id: 'social-scholar',
        title: 'Google Scholar',
        section: 'Socials',
        handler: () => {
          window.open("https://scholar.google.com/citations?user=qc6CJjYAAAAJ", "_blank");
        },
      },{
        id: 'social-custom_social',
        title: 'Custom_social',
        section: 'Socials',
        handler: () => {
          window.open("https://www.alberteinstein.com/", "_blank");
        },
      },{
      id: 'light-theme',
      title: 'Change theme to light',
      description: 'Change the theme of the site to Light',
      section: 'Theme',
      handler: () => {
        setThemeSetting("light");
      },
    },
    {
      id: 'dark-theme',
      title: 'Change theme to dark',
      description: 'Change the theme of the site to Dark',
      section: 'Theme',
      handler: () => {
        setThemeSetting("dark");
      },
    },
    {
      id: 'system-theme',
      title: 'Use system default theme',
      description: 'Change the theme of the site to System Default',
      section: 'Theme',
      handler: () => {
        setThemeSetting("system");
      },
    },];
