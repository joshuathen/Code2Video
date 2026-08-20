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

class Section2Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Vector Addition: The Head-to-Tail Rule", [
            "Vectors add by head-to-tail placement.",
            "Place the second vector's tail at the head.",
            "The resultant is the total displacement."
        ])
        
        # Assets
        pencil = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/pencil.svg")
        ruler = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ruler.svg")
        protractor = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/protractor.svg")
        
        # Define vectors
        u = Vector([1.5, 1, 0], color=WHITE)
        v = Vector([1, -1.5, 0], color=WHITE)
        u_label = MathTex(r"\\vec{u}", color=WHITE)
        v_label = MathTex(r"\\vec{v}", color=WHITE)
        
        # Positioning using fixed grid points
        self.place_at_grid(u, "B4", scale_factor=0.8)
        self.place_at_grid(u_label, "A4", scale_factor=0.7)
        self.place_at_grid(v, "C4", scale_factor=0.8)
        self.place_at_grid(v_label, "D4", scale_factor=0.7)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        self.place_at_grid(pencil, "F6", scale_factor=0.2)
        self.play(FadeIn(pencil))
        self.play(Create(u), Write(u_label), Create(v), Write(v_label))
        self.play(FadeOut(pencil))

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        self.place_at_grid(ruler, "F6", scale_factor=0.2)
        self.play(FadeIn(ruler))
        # Shift v to start at the head of u
        target_pos = u.get_end()
        shift_vec = target_pos - v.get_start()
        self.play(
            v.animate.shift(shift_vec),
            v_label.animate.shift(shift_vec)
        )
        self.play(FadeOut(ruler))

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(GREEN)
        self.place_at_grid(protractor, "F6", scale_factor=0.2)
        self.play(FadeIn(protractor))
        # Resultant vector
        resultant = Vector(u.get_vector() + v.get_vector(), color=GREEN)
        resultant.shift(u.get_start())
        res_label = MathTex(r"\\vec{u}+\\vec{v}", color=GREEN).next_to(resultant.get_center(), RIGHT)
        
        self.play(Create(resultant), Write(res_label))
        self.play(FadeOut(protractor))
        self.wait(2)
