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
        self.setup_layout("The Mathematical Blueprint", [
            "The integral acts like a mathematical sieve.",
            "Multiplication with exponentials isolates specific frequencies.",
            "It filters components for each frequency value."
        ])
        
        # Formula setup
        formula = MathTex(r"e^{i\theta} = \cos(\theta) + i \sin(\theta)", font_size=42)
        
        # Grid visual
        grid_visual = VGroup()
        for pos in self.grid.values():
            grid_visual.add(Dot(pos, radius=0.05, color=GREY))
            
        # Applying requested layout updates
        self.place_in_area(grid_visual, 'A4', 'F6', scale_factor=0.6)
        self.place_at_grid(formula, 'D3', scale_factor=0.9)
        self.add(grid_visual)

        # === Animation for Lecture Line 1 ===
        self.play(Write(formula), run_time=2)
        self.play(self.lecture[0].animate.set_color("#FFFF00"))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Isolate cos term
        cos_term = formula[0][5:13]
        self.play(
            self.lecture[1].animate.set_color("#FFFF00"),
            Indicate(cos_term, color="#FF0000"),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Isolate sin term
        sin_term = formula[0][14:20]
        self.play(
            self.lecture[2].animate.set_color("#FFFF00"),
            Indicate(sin_term, color="#00FF00"),
            run_time=2
        )
        self.wait(2)
