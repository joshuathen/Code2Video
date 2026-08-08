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
        lecture_lines = [
            "Phase space wedges model circular geometry.",
            "Collisions discretize the circular angle.",
            "Mass ratios link physics to pi."
        ]
        self.setup_layout("Conclusion and Intuition", lecture_lines)
        
        # Prepare graphical elements
        phase_space = Sector(radius=1.5, angle=PI/6, color=BLUE).rotate(PI/12)
        theorem = Text("Physics Collision = Pi Expansion", font_size=24, color=WHITE)
        pi_sym = MathTex(r"\\pi", color=YELLOW).scale(3)
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(self.place_at_grid(phase_space, 'B2', scale_factor=0.6)))
        self.lecture[0].set_color("#FFFFFF")

        # === Animation for Lecture Line 2 ===
        self.play(FadeIn(self.place_in_area(theorem, 'A4', 'F6', scale_factor=0.5)))
        self.lecture[1].set_color("#FFFFFF")

        # === Animation for Lecture Line 3 ===
        self.play(ReplacementTransform(theorem, self.place_at_grid(pi_sym, 'D5', scale_factor=0.9)))
        self.lecture[2].set_color("#7FFF00")
        
        self.wait(2)
