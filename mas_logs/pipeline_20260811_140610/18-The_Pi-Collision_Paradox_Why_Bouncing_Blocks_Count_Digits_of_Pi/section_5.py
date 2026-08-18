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
        self.setup_layout("Conclusion and Deep Insight", [
            "Pi emerges from simple physical symmetry.",
            "Conservation laws dictate these complex patterns.",
            "Nature hides Pi in linear motion."
        ])
        
        # --- Animation for Lecture Line 1 ---
        pi_sym = MathTex(r"\\pi", color="#FF00FF", font_size=72)
        # Using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/block.svg]
        block = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/block.svg", color=BLUE, fill_opacity=0.6).scale(0.5)
        self.place_at_grid(pi_sym, 'A2', scale_factor=0.9)
        self.place_at_grid(block, 'A5', scale_factor=0.9)
        
        self.play(FadeIn(pi_sym), FadeIn(block))
        self.play(pi_sym.animate.move_to(block.get_center()), run_time=1.5)
        self.play(self.lecture[0].animate.set_color("#FF00FF"))

        # --- Animation for Lecture Line 2 ---
        formula = MathTex(r"E_k + E_p = \\text{const}", color="#00FFFF")
        arc = Arc(radius=1.0, start_angle=PI/2, angle=-PI, color=YELLOW)
        self.place_at_grid(formula, 'C2', scale_factor=0.7)
        self.place_at_grid(arc, 'C5', scale_factor=0.7)
        
        self.play(Write(formula))
        self.play(Create(arc))
        self.play(FadeOut(formula), FadeOut(arc))
        self.play(self.lecture[1].animate.set_color("#00FFFF"))

        # --- Animation for Lecture Line 3 ---
        # Final wide view showing the full phase space Pi path #FFFFFF including a surrounding frame of [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/block.svg]
        final_text = Text("Nature hides Pi in linear motion.", font_size=36, color="#FFFFFF")
        
        # Frame using blocks
        frame = VGroup()
        for pos in ['D1', 'D6', 'E1', 'E6', 'F1', 'F6']:
            b = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/block.svg", color=WHITE).scale(0.3)
            self.place_at_grid(b, pos)
            frame.add(b)
        
        self.place_in_area(final_text, 'E2', 'F5', scale_factor=0.6)
        
        self.play(FadeIn(final_text, shift=UP), FadeIn(frame))
        self.play(self.lecture[2].animate.set_color("#FFFFFF"))
        self.wait(2)
