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
        self.setup_layout("Conclusion: The Fundamental Trade-off", ["Localization in both domains is impossible.", "This is a fundamental wave property.", "Physics trade-offs are unavoidable constraints."])
        
        # --- Visual Setup ---
        # Represent the seesaw balance of Time vs Frequency
        seesaw_base = Triangle().scale(0.5).set_fill(GREY, 1)
        seesaw_bar = Line(LEFT*2, RIGHT*2, color=WHITE)
        seesaw = VGroup(seesaw_base, seesaw_bar)
        
        # Time Precision (left side) and Frequency Precision (right side)
        time_text = Text("Time Precision", font_size=20, color=BLUE)
        freq_text = Text("Frequency Precision", font_size=20, color=YELLOW)
        
        self.place_in_area(seesaw, 'D2', 'D5', scale_factor=1.1)
        self.place_at_grid(time_text, 'D2', scale_factor=0.7)
        self.place_at_grid(freq_text, 'D5', scale_factor=0.7)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(BLUE))
        self.play(Write(seesaw), Write(time_text), Write(freq_text))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(YELLOW))
        # Animate the seesaw tilting to show inverse relationship
        self.play(Rotate(seesaw_bar, angle=PI/8, about_point=seesaw_bar.get_center()),
                  time_text.animate.shift(UP*0.5),
                  freq_text.animate.shift(DOWN*0.5))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(GREEN))
        # Stabilize
        self.play(Rotate(seesaw_bar, angle=-PI/8, about_point=seesaw_bar.get_center()),
                  time_text.animate.shift(DOWN*0.5),
                  freq_text.animate.shift(UP*0.5))
        self.wait(2)
