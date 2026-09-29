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

class Section5Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Synthesis & Summary", [
            "Domain coloring creates the visual landscape.",
            "Winding numbers provide the mathematical audit.",
            "Visual tools solve complex algebraic problems."
        ])
        
        # Assets
        notebook = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/notebook.svg")
        magnifying_glass = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/magnifying-glass.svg")
        
        # Elements
        landscape = Square(side_length=2, color=BLUE_D, fill_opacity=0.3)
        audit = Circle(radius=1, color=YELLOW_D)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(BLUE_C))
        self.place_in_area(landscape, "D1", "F3", scale_factor=0.8)
        self.place_at_grid(notebook, "A1", scale_factor=0.5)
        self.play(Create(landscape), FadeIn(notebook))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(YELLOW_C))
        self.place_at_grid(audit, "D5", scale_factor=0.6)
        self.place_at_grid(magnifying_glass, "B5", scale_factor=0.5)
        self.play(Create(audit), FadeIn(magnifying_glass))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(GREEN_C))
        visual_tools = VGroup(landscape, audit, notebook, magnifying_glass)
        self.place_in_area(visual_tools, "A4", "C6", scale_factor=0.7)
        self.play(visual_tools.animate.set_opacity(0.5))
        
        self.wait(2)
        self.play(FadeOut(self.lecture), FadeOut(self.title), FadeOut(visual_tools))
