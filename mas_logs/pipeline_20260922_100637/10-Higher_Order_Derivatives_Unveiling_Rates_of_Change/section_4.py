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
        lecture_lines = ["Third derivative measures the rate of acceleration change.", 
                         "In physics, this value is called jerk.", 
                         "High jerk creates rigid, robotic movement."]
        self.setup_layout("Higher Orders (3rd and Beyond)", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        f_triple = MathTex(r"f'''(x) = \frac{d}{dx} a(x)", color="#FFA500")
        self.place_at_grid(f_triple, 'B2', scale_factor=1.0)
        self.play(Write(f_triple))
        self.play(self.lecture[0].animate.set_color("#FFA500"))

        # === Animation for Lecture Line 2 ===
        jerk_text = Text("Jerk", color="#FFFF00")
        self.place_at_grid(jerk_text, 'B3', scale_factor=1.2)
        self.play(FadeIn(jerk_text))
        self.play(self.lecture[1].animate.set_color("#FFFF00"))

        # === Animation for Lecture Line 3 ===
        # Using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg]
        robotic_arm = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg")
        robotic_arm.set_color("#00FFFF")
        self.place_in_area(robotic_arm, 'D5', 'F6', scale_factor=0.8)
        
        self.play(FadeIn(robotic_arm))
        self.play(
            robotic_arm.animate.rotate(0.5, about_point=robotic_arm.get_center()),
            run_time=1, rate_func=linear
        )
        self.play(self.lecture[2].animate.set_color("#00FFFF"))
        self.wait(2)
