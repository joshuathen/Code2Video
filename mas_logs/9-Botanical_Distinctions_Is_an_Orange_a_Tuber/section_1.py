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
        # 1. Setup layout with Section Title and Lecture Lines
        lecture_lines = [
            "Oliver finds an orange and a potato nearby.",
            "Both are round and grow from plants.",
            "Could an orange actually be a tuber?"
        ]
        self.setup_layout("Introduction: The Botanical Mystery", lecture_lines)

        # 2. Define Assets and Objects
        # Oliver the Owl (#FFFFFF) [Asset: Oliver the Owl]
        oliver = Circle(color=BLUE, fill_opacity=0.5) 
        self.place_in_area(oliver, "C3", "D4", scale_factor=1.8)

        # Orange and Potato objects
        # Fixed overlap issues (Issue 25, 26)
        orange_shape = Circle(radius=0.45, color="#FFA500", fill_opacity=1)
        self.place_at_grid(orange_shape, "C1", scale_factor=1.0)

        potato_shape = Ellipse(width=1.0, height=0.7, color="#D2B48C", fill_opacity=1)
        self.place_at_grid(potato_shape, "C6", scale_factor=1.0)

        # Question Mark for the final hook
        # Fixed overlap issue (Issue 27)
        question_mark = Text("?", font_size=80, color="#FFFF00")
        self.place_in_area(question_mark, "A3", "A4", scale_factor=0.8)

        # === Animation for Lecture Line 1 ===
        # Oliver finds an orange and a potato nearby.
        self.play(
            FadeIn(oliver),
            self.lecture[0].animate.set_color("#FFA500"),
            run_time=1
        )
        self.play(
            FadeIn(orange_shape),
            FadeIn(potato_shape),
            run_time=1
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Both are round and grow from plants.
        self.play(
            self.lecture[1].animate.set_color("#A9A9A9"),
            run_time=1
        )
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # Could an orange actually be a tuber?
        self.play(
            FadeIn(question_mark),
            self.lecture[2].animate.set_color("#FFFF00"),
            run_time=1
        )
        self.wait(3)
