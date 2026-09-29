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
        self.setup_layout("Scaling: Changing Magnitude", [
            "Scaling changes a vector's magnitude.",
            "Multiply by k to stretch or shrink.",
            "Negative k values reverse the direction."
        ])
        
        # Define mobjects
        v_start = ORIGIN
        v_end = RIGHT * 1.5 + UP * 1.5
        
        vec_v = Arrow(v_start, v_end, color="#FF00FF", buff=0)
        label_v = MathTex("v", color="#FF00FF")
        
        vec_2v = Arrow(v_start, v_end * 2, color="#00FF00", buff=0)
        label_2v = MathTex("2v", color="#00FF00")
        
        vec_05v = Arrow(v_start, v_end * 0.5, color="#FFFF00", buff=0)
        label_05v = MathTex("0.5v", color="#FFFF00")
        
        vec_neg_v = Arrow(v_start, v_end * -1, color="#00FFFF", buff=0)
        label_neg_v = MathTex("-v", color="#00FFFF")

        # === Animation for Lecture Line 1 ===
        self.place_in_area(VGroup(vec_v, label_v), 'A3', 'F5', scale_factor=1.0)
        label_v.next_to(vec_v.get_end(), UR, buff=0.1)
        
        self.play(Create(vec_v), Write(label_v))
        self.play(self.lecture[0].animate.set_color("#FF00FF"))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.place_at_grid(label_2v, 'B4', scale_factor=0.9)
        self.place_at_grid(label_05v, 'D4', scale_factor=0.9)
        
        # 2v
        self.play(ReplacementTransform(vec_v.copy(), vec_2v), FadeIn(label_2v))
        self.play(self.lecture[1].animate.set_color("#00FF00"))
        self.wait(1)
        
        # 0.5v
        self.play(ReplacementTransform(vec_v.copy(), vec_05v), FadeIn(label_05v))
        self.play(self.lecture[1].animate.set_color("#FFFF00"))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        negative_vector_group = VGroup(vec_neg_v, label_neg_v)
        self.place_in_area(negative_vector_group, 'C3', 'E5', scale_factor=0.8)
        
        self.play(Create(vec_neg_v), Write(label_neg_v))
        self.play(self.lecture[2].animate.set_color("#00FFFF"))
        self.wait(2)
