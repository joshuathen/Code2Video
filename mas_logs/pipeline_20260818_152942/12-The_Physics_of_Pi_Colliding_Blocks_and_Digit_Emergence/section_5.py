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
            "Geometry maps reflection count to pi digits.",
            "Higher mass ratios reveal more digits.",
            "Physics truly connects to mathematics.",
            "It's an elegant, hidden geometric law.",
            "A beautiful realization of pi."
        ]
        self.setup_layout("Conclusion: Why Pi?", lecture_lines)
        
        # Prep assets
        pi_text = Text("π ≈ 3.14159...", font_size=40, color="#F8F9FA")
        geometry_group = VGroup(
            Arc(radius=1.0, angle=PI/2, color="#FFC107"),
            Line(ORIGIN, RIGHT, color="#FFC107"),
            Line(ORIGIN, UP, color="#FFC107")
        )
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#F8F9FA")
        self.play(FadeIn(self.place_at_grid(pi_text, 'C4', scale_factor=1.0)))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#F8F9FA")
        self.play(FadeIn(self.place_in_area(geometry_group, 'D2', 'F4', scale_factor=0.9)))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#F8F9FA")
        geometry_group.set_color("#FFC107")
        self.play(Indicate(geometry_group))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#F8F9FA")
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#F8F9FA")
        self.play(FadeOut(pi_text), FadeOut(geometry_group))
        self.wait(2)
