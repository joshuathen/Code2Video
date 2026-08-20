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
        lecture_lines = [
            "Outbreaks spread through population contact.",
            "People fall into three health states.",
            "Susceptible: healthy, yet to be exposed.",
            "Infected: currently carrying and spreading disease.",
            "Recovered: immune or no longer infectious."
        ]
        self.setup_layout("Introduction: Why Do Outbreaks Spread?", lecture_lines)
        
        # Create visual representations for S, I, and R
        s_dot = Dot(color=BLUE, radius=0.2)
        i_dot = Dot(color=RED, radius=0.2)
        r_dot = Dot(color=GREEN, radius=0.2)
        
        # Labels
        s_text = Text("S", font_size=24, color=BLUE)
        i_text = Text("I", font_size=24, color=RED)
        r_text = Text("R", font_size=24, color=GREEN)
        
        # Group them
        s_group = VGroup(s_dot, s_text).arrange(RIGHT)
        i_group = VGroup(i_dot, i_text).arrange(RIGHT)
        r_group = VGroup(r_dot, r_text).arrange(RIGHT)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(YELLOW))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(BLUE))
        self.place_at_grid(s_group, 'B2')
        self.play(FadeIn(s_group))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(RED))
        self.place_at_grid(i_group, 'C2')
        self.play(FadeIn(i_group))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(GREEN))
        self.place_at_grid(r_group, 'D2')
        self.play(FadeIn(r_group))
        self.wait(2)
