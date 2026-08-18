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
            "Cheetahs show us changing motion clearly.",
            "Velocity is the derivative of position.",
            "Acceleration is the derivative of velocity.",
            "Rates of change build a chain.",
            "Each step describes motion better."
        ]
        self.setup_layout("Intuitive Hook: The Running Cheetah", lecture_lines)
        
        # Cheetah
        cheetah = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cheetah.svg", color="#FF8C00")
        cheetah_label = Text("Cheetah", font_size=20)
        cheetah_group = VGroup(cheetah, cheetah_label).arrange(DOWN)
        self.place_at_grid(cheetah_group, 'C5', scale_factor=0.6)
        
        # Vectors
        velocity_arrow = Arrow(start=ORIGIN, end=RIGHT*1.5, color="#00FFFF")
        velocity_label = Text("Velocity", font_size=18, color="#00FFFF")
        velocity_group = VGroup(velocity_arrow, velocity_label).arrange(DOWN)
        
        accel_arrow = Arrow(start=ORIGIN, end=RIGHT*1, color="#FFFF00")
        accel_label = Text("Accel", font_size=18, color="#FFFF00")
        accel_group = VGroup(accel_arrow, accel_label).arrange(DOWN)

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(cheetah_group))
        self.lecture[0].set_color("#FF8C00")

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[0].animate.set_color(WHITE), self.lecture[1].animate.set_color("#00FFFF"))
        # Placing velocity arrow according to critic
        self.place_in_area(velocity_arrow, 'B4', 'B6', scale_factor=0.5)
        self.play(GrowArrow(velocity_arrow), Write(velocity_label))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[1].animate.set_color(WHITE), self.lecture[2].animate.set_color("#FFFF00"))
        # Using fix for acceleration label
        self.place_at_grid(accel_label, 'D5', scale_factor=0.7)
        accel_arrow.next_to(cheetah, DOWN)
        self.play(GrowArrow(accel_arrow), Write(accel_label))

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[2].animate.set_color(WHITE), self.lecture[3].animate.set_color(BLUE))
        self.play(Indicate(velocity_arrow), Indicate(accel_arrow))

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[3].animate.set_color(WHITE), self.lecture[4].animate.set_color(GREEN))
        self.play(cheetah.animate.shift(RIGHT*1), velocity_arrow.animate.shift(RIGHT*1), accel_arrow.animate.shift(RIGHT*1))
        self.wait(2)
