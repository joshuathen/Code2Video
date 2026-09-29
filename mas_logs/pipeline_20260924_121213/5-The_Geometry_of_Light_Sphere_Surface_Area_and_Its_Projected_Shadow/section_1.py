from manim import *

class TeachingScene(Scene):
    def setup_layout(self, title_text, lecture_lines):
        # BASE
        self.camera.background_color = "#000000"
        self.title = Text(title_text, font_size=28, color=WHITE).to_edge(UP)
        self.add(self.title)

        # Left-side lecture content (bullets with "-")
        lecture_texts = [Text(line, font_size=22, color=WHITE) for line in lecture_lines]
        self.lecture = VGroup(*lecture_texts).arrange(DOWN, aligned_edge=LEFT).scale(0.8)
        self.lecture.to_edge(LEFT, buff=0.2)
        self.add(self.lecture)

        # Define fine-grained animation grid (4x4 grid on right side)
        self.grid = {}
        rows = ["A", "B", "C", "D", "E", "F"]  # Top to bottom
        cols = ["1", "2", "3", "4", "5", "6"]  # Left to right

        for i, row in enumerate(rows):
            for j, col in enumerate(cols):
                x = 0.5 + j * 1
                y = 2.2 - i * 1
                self.grid[f"{row}{col}"] = np.array([x, y, 0])

    def place_at_grid(self, mobject, grid_pos, scale_factor=1.0):
        mobject.scale(scale_factor)
        mobject.move_to(self.grid[grid_pos])
        return mobject

    def place_in_area(self, mobject, top_left, bottom_right, scale_factor=1.0):
        tl_pos = self.grid[top_left]
        br_pos = self.grid[bottom_right]
        
        # Calculate center of the area
        center_x = (tl_pos[0] + br_pos[0]) / 2
        center_y = (tl_pos[1] + br_pos[1]) / 2
        center = np.array([center_x, center_y, 0])
        
        mobject.scale(scale_factor)
        mobject.move_to(center)
        return mobject

class Section1Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Prerequisite Visualization: The Circle vs. The Sphere", 
                          ["A sphere has a constant radius from its center.", 
                           "Circles are 2D; spheres are 3D objects.", 
                           "Parallel light casts a circular shadow on planes."])
        
        # Assets
        planet_svg = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/planet.svg")
        sun_svg = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sun.svg")

        # === Animation for Lecture Line 1 ===
        # A sphere has a constant radius from its center.
        self.lecture[0].set_color("#ADD8E6")
        
        # Using asset as a stand-in for the circle representation
        circle = planet_svg.copy()
        circle.set_color("#ADD8E6")
        self.place_at_grid(circle, 'B2', scale_factor=0.6)
        
        radius_line = Line(circle.get_center(), circle.get_center() + RIGHT * 0.8, color="#FFFFFF")
        r_label = Tex("r", color="#FFFFFF")
        self.place_at_grid(r_label, 'B3', scale_factor=0.9)
        
        self.play(FadeIn(circle))
        self.play(Create(radius_line), Write(r_label))
        
        # === Animation for Lecture Line 2 ===
        # Circles are 2D; spheres are 3D objects.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#ADD8E6")
        
        # Transition to Sphere (3D)
        sphere = Sphere(radius=0.6, color="#ADD8E6")
        self.place_at_grid(sphere, 'B2')
        
        self.play(Transform(circle, sphere))
        
        # === Animation for Lecture Line 3 ===
        # Parallel light casts a circular shadow on planes.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#ADD8E6")
        
        shadow = Circle(radius=0.6, color="#FFD700", fill_opacity=0.5)
        self.place_at_grid(shadow, 'E2', scale_factor=1.0)
        
        # Highlight cross-section using Sun icon
        sun_highlight = sun_svg.copy()
        sun_highlight.set_color("#FFD700")
        self.place_at_grid(sun_highlight, 'E5', scale_factor=0.4)
        
        label = Text("2D Shadow", font_size=24, color='#FFD700')
        self.place_at_grid(label, 'E3')
        self.add(label)
        
        self.play(Create(shadow), FadeIn(sun_highlight))
        self.wait(1)
