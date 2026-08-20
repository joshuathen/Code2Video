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

class Section3Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Vector Addition (The Tip-to-Tail Rule)", [
            "Adding vectors follows simple rules.",
            "Connect the tail to the tip.",
            "The result is the final path."
        ])
        
        # Define vectors
        v = Vector(RIGHT*1.5 + UP*0.5, color=WHITE)
        w = Vector(RIGHT*0.5 + UP*1.5, color=WHITE)
        
        v_label = MathTex(r"\\vec{v}", color=WHITE).next_to(v.get_center(), UP)
        w_label = MathTex(r"\\vec{w}", color=WHITE).next_to(w.get_center(), RIGHT)

        # Assets
        ruler = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ruler.svg")
        pencil = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/pencil.svg")

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#00FFFF"))
        self.place_at_grid(ruler, 'B6', scale_factor=0.3)
        self.place_at_grid(v, 'C3', scale_factor=0.9)
        self.place_at_grid(w, 'D4', scale_factor=0.9)
        self.play(FadeIn(ruler), Create(v), Write(v_label), Create(w), Write(w_label))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FFFF00"))
        
        # Move w so its tail meets the tip of v
        new_w_start = v.get_end()
        w_vec = w.get_end() - w.get_start()
        w_new_pos = new_w_start + w_vec / 2
        
        self.play(
            w.animate.move_to(w_new_pos),
            w_label.animate.move_to(w_new_pos + RIGHT*0.5 + UP*0.5)
        )

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FF0000"))
        
        resultant = Vector(v.get_vector() + w.get_vector(), color=RED)
        resultant.move_to(v.get_start() + resultant.get_vector()/2)
        res_label = MathTex(r"\\vec{v}+\\vec{w}", color=RED)
        self.place_in_area(res_label, 'E2', 'F4', scale_factor=0.8)
        self.place_at_grid(pencil, 'F5', scale_factor=0.3)
        
        self.play(Create(resultant), Write(res_label), FadeIn(pencil))
        self.wait(2)
