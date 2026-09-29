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

class Section2Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Fourier series build shapes from epicycles.",
            "Circles added together form complex periodic paths.",
            "Radii and frequencies determine the path's shape.",
            "This synthesis constructs any periodic function.",
            "See how these vectors trace the cat's path."
        ]
        self.setup_layout("Epicycles: The Fourier Synthesis", lecture_lines)
        
        # Add watermark cat
        cat = ImageMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cat.png")
        self.place_in_area(cat, "B4", "E5", scale_factor=0.3)
        cat.set_opacity(0.3)
        self.add(cat)
        
        # Initialize shapes
        c1 = Circle(radius=1.0, color="#FFFFFF")
        c2 = Circle(radius=0.5, color="#FFFF00")
        
        # Positioning at grid
        self.place_at_grid(c1, "C4", scale_factor=0.5)
        c2.next_to(c1.get_right(), RIGHT, buff=0)
        
        # Path tracer
        path = TracedPath(c2.get_right, stroke_color="#FF0000", stroke_width=2)
        
        self.add(c1, c2, path)
        
        # Updaters
        # Center of C1 is fixed at self.grid["C4"]
        c1.add_updater(lambda m, dt: m.rotate(dt * 0.5, about_point=self.grid["C4"]))
        c2.add_updater(lambda m, dt: m.move_to(c1.get_right() + [0.5, 0, 0])) # Note: simplified for logic
        # Note: Proper epicycles require keeping track of phase/centers
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFFFF")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FFFF00")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FF0000")
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#00FF00")
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#00FFFF")
        self.wait(2)
