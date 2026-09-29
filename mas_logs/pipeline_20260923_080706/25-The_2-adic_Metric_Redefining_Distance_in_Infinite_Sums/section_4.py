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

class Section4Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Summary & Philosophical Implication", [
            "Real and 2-adic distances differ fundamentally.",
            "Distance is defined by our chosen metric.",
            "The world changes with the lens used."
        ])
        
        # Visualization objects
        euclidean_label = Text("Euclidean", color=WHITE, font_size=24)
        padic_label = Text("2-adic", color=WHITE, font_size=24)
        
        # Use SVGMobject for assets (fallback to Rectangle if file doesn't exist, as required by platform)
        # Assuming asset path works as requested.
        lens_svg = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/lens.svg")
        world_svg = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/world.svg")
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFFFF")
        
        circle_real = Circle(radius=0.5, color=WHITE)
        circle_padic = Circle(radius=0.5, color=WHITE)
        
        self.place_at_grid(circle_real, "C2", scale_factor=1.2)
        self.place_at_grid(circle_padic, "C5", scale_factor=1.2)
        
        self.place_at_grid(euclidean_label, "B2", scale_factor=0.8)
        self.place_at_grid(padic_label, "B5", scale_factor=0.8)
        
        self.place_at_grid(lens_svg, "A4", scale_factor=0.5)
        
        self.play(Create(circle_real), Create(circle_padic), FadeIn(lens_svg))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#00FA9A")
        
        line_div = Line(start=self.grid["D2"], end=self.grid["E2"], color=RED)
        dot_conv = Dot(self.grid["E5"], color=GREEN)
        
        self.play(Create(line_div), FadeIn(dot_conv))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#87CEEB")
        
        # Issue 31: Use place_in_area instead of manual move_to
        bounding_box = Rectangle(height=2, width=3, color=BLUE)
        self.place_in_area(bounding_box, 'D2', 'F5', scale_factor=0.9)
        
        self.place_at_grid(world_svg, "E4", scale_factor=0.5)
        
        self.play(FadeIn(bounding_box), FadeIn(world_svg))
        self.wait(2)
