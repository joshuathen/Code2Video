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

class Section2Scene(TeachingScene):
    def construct(self):
        # Title and Lecture Lines for Section 2
        title_text = "Prerequisite: Dot Products as Similarity"
        lecture_lines = [
            "How does the model \"search\" for a specific fact?",
            "Dot products measure similarity between two vectors.",
            "High similarity triggers the retrieval of stored information."
        ]
        self.setup_layout(title_text, lecture_lines)
        
        # Define base coordinates
        origin_point = self.grid["D3"]
        vec_a_end = self.grid["B3"]
        vec_b_start = self.grid["D5"]
        
        # === Animation for Lecture Line 1 ===
        # Display two vectors 'a' and 'b' at 90-degree angle (#FFFFFF).
        self.lecture[0].set_color(WHITE)
        
        vec_a = Arrow(origin_point, vec_a_end, buff=0, color=WHITE)
        vec_b = Arrow(origin_point, vec_b_start, buff=0, color=WHITE)
        
        label_a = MathTex("a", color=WHITE)
        self.place_at_grid(label_a, "B3")
        label_a.shift(UP * 0.4)
        
        label_b = MathTex("b", color=WHITE)
        self.place_at_grid(label_b, "D5")
        label_b.shift(RIGHT * 0.4)
        
        self.play(
            Create(vec_a),
            Create(vec_b),
            Write(label_a),
            Write(label_b),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Rotate vector 'b' to align with 'a' and turn green (#00FF00).
        self.lecture[0].set_color(GRAY)
        self.lecture[1].set_color("#00FF00")
        
        angle_tracker = ValueTracker(0) # 0 (90 deg) to PI/2 (0 deg diff)
        
        def get_b_end():
            angle = angle_tracker.get_value()
            # Initial relative vector from D3 to D5 is [2, 0, 0]
            rel_vec = np.array([2, 0, 0])
            rot_mat = rotation_matrix(angle, OUT)
            rotated = np.dot(rot_mat, rel_vec)
            return origin_point + rotated

        # Updaters for smooth transition
        vec_b.add_updater(lambda m: m.put_start_and_end_on(
            origin_point, get_b_end()
        ).set_color(interpolate_color(WHITE, "#00FF00", angle_tracker.get_value() / (PI/2))))
        
        label_b.add_updater(lambda m: m.move_to(
            get_b_end() + RIGHT * 0.3 + UP * 0.2
        ).set_color(interpolate_color(WHITE, "#00FF00", angle_tracker.get_value() / (PI/2))))
        
        self.play(
            angle_tracker.animate.set_value(PI/2),
            run_time=2.5,
            rate_func=smooth
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Show a 'Match Score' bar filling up as alignment increases (#00FF00).
        self.lecture[1].set_color(GRAY)
        self.lecture[2].set_color("#00FF00")
        
        # Match Score Bar container at F4
        bar_bg = Rectangle(height=0.3, width=3.0, color=WHITE).set_stroke(width=2)
        self.place_at_grid(bar_bg, "F4")
        
        # Filling indicator
        bar_fill = Rectangle(height=0.3, width=3.0, color="#00FF00", fill_opacity=0.8).set_stroke(width=0)
        bar_fill.align_to(bar_bg, LEFT)
        
        score_label = Text("Match Score", font_size=20, color=WHITE)
        self.place_at_grid(score_label, "F2")
        # Shift to avoid overlap with bar starting at F3.5-1.5=2.0
        score_label.shift(RIGHT * 0.3) 
        
        fill_tracker = ValueTracker(0)
        bar_fill.add_updater(lambda m: m.stretch_to_fit_width(
            max(0.0001, fill_tracker.get_value() * 3.0), about_edge=LEFT
        ))
        
        self.play(
            Create(bar_bg),
            Write(score_label),
            run_time=1
        )
        
        self.add(bar_fill)
        self.play(
            fill_tracker.animate.set_value(1.0),
            run_time=2,
            rate_func=smooth
        )
        
        # Visual feedback for retrieval trigger
        self.play(
            vec_a.animate.scale(1.1),
            vec_b.animate.scale(1.1),
            rate_func=there_and_back,
            run_time=1
        )
        
        self.wait(2)
        
        # Removal of updaters for clean exit
        vec_b.clear_updaters()
        label_b.clear_updaters()
        bar_fill.clear_updaters()
