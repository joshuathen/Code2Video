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
        lecture_lines = ["Scalars stretch or shrink vectors.", "Direction stays same if positive.", "Negative scalars reverse the direction."]
        self.setup_layout("Scalar Multiplication & Scaling", lecture_lines)
        
        # Define base vector and assets
        # Icons
        ruler = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ruler.svg")
        spring = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/spring.svg")
        magnet = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/magnet.svg")
        
        v = Arrow(start=ORIGIN, end=RIGHT*2, color="#00FFFF")
        v_label = MathTex(r"\\vec{v}", color="#00FFFF").next_to(v, UP)
        v_group = VGroup(v, v_label, ruler)
        self.place_in_area(ruler, "A2", "B2", scale_factor=0.3)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#00FFFF"))
        self.place_in_area(v_group, "A2", "B4", scale_factor=0.8)
        self.play(Create(v), Write(v_label), FadeIn(ruler))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FFFF00"))
        # Scale v by 2
        v2 = Arrow(start=ORIGIN, end=RIGHT*4, color="#FFFF00")
        v2_label = MathTex(r"2\\vec{v}", color="#FFFF00").next_to(v2, UP)
        v2_group = VGroup(v2, v2_label, spring)
        self.place_in_area(spring, "A5", "B5", scale_factor=0.3)
        self.place_in_area(v2_group, "C2", "D4", scale_factor=0.8)
        self.play(ReplacementTransform(v, v2), ReplacementTransform(v_label, v2_label), FadeIn(spring))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FF0000"))
        # Scale v by -1
        v3 = Arrow(start=ORIGIN, end=LEFT*2, color="#FF0000")
        v3_label = MathTex(r"-1\\vec{v}", color="#FF0000").next_to(v3, UP)
        v3_group = VGroup(v3, v3_label, magnet)
        self.place_in_area(magnet, "D5", "E5", scale_factor=0.3)
        self.place_in_area(v3_group, "E2", "F4", scale_factor=0.8)
        self.play(ReplacementTransform(v2, v3), ReplacementTransform(v2_label, v3_label), FadeIn(magnet))
        self.wait(1)
