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
        self.setup_layout("Scalar Multiplication and Scaling", [
            "- Scalar multiplication scales vector length.",
            "- Positive scalars preserve original direction.",
            "- Double speed boosts length to 2v."
        ])
        
        # 1. Vector v setup (Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/car.png)
        car = ImageMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/car.png").scale(0.25)
        v = Arrow(start=LEFT*0.75, end=RIGHT*0.75, color="#00CED1", buff=0)
        v_label = MathTex(r"\\vec{v}", color="#00CED1", font_size=24)
        
        # Group for consistency (use Group instead of VGroup for non-VMobject elements like ImageMobject)
        v_group = Group(car, v, v_label).arrange(DOWN, buff=0.1)
        
        # 2. Vector 2v setup (Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/speedometer.svg)
        speedometer = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/speedometer.svg").scale(0.2)
        v_2 = Arrow(start=LEFT*1.5, end=RIGHT*1.5, color="#FFD700", buff=0)
        v_2_label = MathTex(r"2\\vec{v}", color="#FFD700", font_size=24)
        
        # Group for consistency
        v_2_group = Group(speedometer, v_2, v_2_label).arrange(DOWN, buff=0.1)

        # --- Animations ---
        # === Animation for Lecture Line 1 ===
        # Using place_in_area for better spatial distribution and grid utilization (B3-D5)
        self.place_in_area(v_group, 'B3', 'D5', scale_factor=0.9)
        self.play(FadeIn(v_group))
        self.lecture[0].set_color("#00CED1")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FFD700")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.place_in_area(v_2_group, 'B3', 'D5', scale_factor=0.9)
        self.play(ReplacementTransform(v_group, v_2_group))
        self.lecture[2].set_color("#FFD700")
        self.wait(2)