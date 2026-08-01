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
        lecture_lines = [\
            "DP-3T minimizes data collection.",\
            "Only necessary data is shared.",\
            "This balances public health and privacy."\
        ]
        self.setup_layout("The Privacy Guarantee: Data Minimization and Anonymity", lecture_lines)
        
        # Define colors for lecture lines and corresponding animations
        color1 = "#FFD700"  # Gold
        color2 = "#ADFF2F"  # GreenYellow
        color3 = "#87CEFA"  # LightSkyBlue

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(color1))
        data_minimization_text = Text("Data Minimization", font_size=32, color=color1)
        self.place_at_grid(data_minimization_text, 'B4') # Addressed Issue 35
        self.play(FadeIn(data_minimization_text))
        self.wait(1.5)
        self.play(FadeOut(data_minimization_text))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(color2))
        user_control_text = Text("User Control", font_size=32, color=color2)
        self.place_at_grid(user_control_text, 'C2') # Addressed Issue 34
        self.play(FadeIn(user_control_text))
        self.wait(1.5)
        self.play(FadeOut(user_control_text))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(color3))
        anonymity_ensured_text = Text("Anonymity Ensured", font_size=32, color=color3)
        self.place_at_grid(anonymity_ensured_text, 'B5') # Addressed Issue 36
        self.play(FadeIn(anonymity_ensured_text))
        self.wait(1.5)
        self.play(FadeOut(anonymity_ensured_text))
