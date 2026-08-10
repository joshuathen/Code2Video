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
        lecture_lines = ["Scalars change vector magnitude.", "Positive scalars keep the same direction.", "Negative scalars reverse the direction."]
        self.setup_layout("Scalar Multiplication and Scaling", lecture_lines)
        
        # Setup Assets
        compass = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/compass.svg")
        spring = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/spring.svg")
        
        # Setup Mobjects
        vector = Arrow(start=ORIGIN, end=RIGHT*1.5, color="#90EE90")
        vector_label = MathTex(r"\vec{v}", color="#90EE90")
        v_group = VGroup(vector, vector_label).arrange(UP, buff=0.1)
        
        # Initial state: Position in area avoiding lecture
        self.place_in_area(v_group, "C2", "D4", scale_factor=1.0)
        self.place_at_grid(compass, "B2", scale_factor=0.6)
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(v_group), FadeIn(compass))
        self.play(self.lecture[0].animate.set_color("#90EE90"))
        
        # === Animation for Lecture Line 2 ===
        c = MathTex("c = 2", color="#FFFF00")
        self.place_at_grid(c, "A3", scale_factor=1.0) # Fixed via Issue 29
        
        # Create scaled version
        vector_scaled = Arrow(start=ORIGIN, end=RIGHT*3, color="#FFFF00")
        vector_scaled_label = MathTex(r"2\vec{v}", color="#FFFF00")
        v_scaled_group = VGroup(vector_scaled, vector_scaled_label).arrange(UP, buff=0.1)
        
        # Fix animation placement via Issue 31
        self.play(Write(c))
        self.play(self.lecture[1].animate.set_color("#FFFF00"))
        self.place_in_area(v_scaled_group, "D3", "F6", scale_factor=0.8) # Fixed via Issue 31
        self.play(ReplacementTransform(v_group, v_scaled_group))
        
        # === Animation for Lecture Line 3 ===
        c_neg = MathTex(r"c = -1", color="#FF0000")
        self.place_at_grid(c_neg, "B3", scale_factor=1.0) # Fixed via Issue 30
        self.place_at_grid(spring, "B4", scale_factor=0.5)
        
        vector_neg = Arrow(start=ORIGIN, end=LEFT*1.5, color="#FF0000")
        vector_neg_label = MathTex(r"-1\vec{v}", color="#FF0000")
        v_neg_group = VGroup(vector_neg, vector_neg_label).arrange(UP, buff=0.1)
        
        self.play(ReplacementTransform(c, c_neg), FadeIn(spring))
        self.play(self.lecture[2].animate.set_color("#FF0000"))
        self.place_in_area(v_neg_group, "D3", "F6", scale_factor=0.8)
        self.play(ReplacementTransform(v_scaled_group, v_neg_group))
        
        self.wait(2)
