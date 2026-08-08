from manim import *
import numpy as np

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
        # Setup title and lecture lines
        title_str = "The Hook: The Cheetah's Dash"
        lecture_lines = [
            "Swift the cheetah runs 100 meters in 4 seconds.",
            "His average speed is exactly 25 meters per second.",
            "But how fast is he at one specific moment?"
        ]
        self.setup_layout(title_str, lecture_lines)

        # Colors for elements
        GRAY_COLOR = "#D3D3D3"
        WHITE_COLOR = "#FFFFFF"
        YELLOW_COLOR = "#FFFF00"

        # === Animation for Lecture Line 1 ===
        # Line: "Swift the cheetah runs 100 meters in 4 seconds."
        self.lecture[0].set_color(GRAY_COLOR)
        
        # Track path from B3 to B6 (Column 3 to 6 to respect B021)
        start_coord = self.grid["B3"]
        end_coord = self.grid["B6"]
        track = Line(start_coord, end_coord, color=GRAY_COLOR)
        
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/cheetah.svg]
        # Load cheetah asset as a persistent mobject
        swift = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cheetah.svg")
        swift.set_color(WHITE_COLOR)
        self.place_at_grid(swift, "B3", scale_factor=0.5)
        
        # Persistent label for Swift
        swift_label = Text("Swift", font_size=16, color=WHITE_COLOR)
        # B008: Use updater for position instead of recreating
        swift_label.add_updater(lambda m: m.next_to(swift, UP, buff=0.1))
        
        # Descriptive labels
        dist_label = Text("100 meters", font_size=20, color=GRAY_COLOR)
        self.place_at_grid(dist_label, "C4")
        
        time_label = Text("4 seconds", font_size=20, color=GRAY_COLOR)
        # Issue 28: Move time_label to D4 to avoid horizontal overlap with dist_label
        self.place_at_grid(time_label, "D4")
        
        self.play(Create(track), Write(dist_label), Write(time_label))
        self.add(swift, swift_label)
        
        # Swift runs the distance in 4 seconds
        self.play(
            swift.animate.move_to(end_coord),
            run_time=4,
            rate_func=linear
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Line: "His average speed is exactly 25 meters per second."
        self.lecture[1].set_color(WHITE_COLOR)
        
        avg_speed_text = Text("Average Speed = 25m/s", font_size=22, color=WHITE_COLOR)
        # Issue 29: Use place_in_area from A3 to A5 to prevent overlap with the question mark
        self.place_in_area(avg_speed_text, "A3", "A5", scale_factor=0.8)
        
        self.play(Write(avg_speed_text))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # Line: "But how fast is he at one specific moment?"
        self.lecture[2].set_color(YELLOW_COLOR)
        
        # Freeze Swift at 2.5 seconds (0.625 of the total 4.0s path)
        freeze_ratio = 2.5 / 4.0
        freeze_point = start_coord + (end_coord - start_coord) * freeze_ratio
        
        # Highlight and move Swift to the frozen moment
        self.play(
            swift.animate.move_to(freeze_point).set_color(YELLOW_COLOR),
            run_time=1
        )
        
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/question.svg]
        # Large question mark to represent the core inquiry
        q_mark = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/question.svg")
        q_mark.set_color(YELLOW_COLOR)
        # Issue 30: Move question mark to B5 to avoid overlap with avg_speed_text at A5
        self.place_at_grid(q_mark, "B5", scale_factor=0.6)
        
        self.play(FadeIn(q_mark, scale=1.2))
        self.wait(3)
