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
        self.setup_layout("Linear Dependence: Redundancy", ["Dependence reveals redundant directions.", "A third vector adds no reach.", "Like hikers following one trail."])
        
        # Setup vectors
        u = Arrow(start=ORIGIN, end=RIGHT*1.5, color=BLUE)
        v = Arrow(start=ORIGIN, end=UP*1.5, color=GREEN)
        w = Arrow(start=ORIGIN, end=(RIGHT+UP)*1.5, color=RED)
        
        # Group them as per advice B002
        vectors_group = VGroup(u, v, w)
        self.place_in_area(vectors_group, 'C3', 'E5', scale_factor=0.9)
        
        # Labels
        u_label = MathTex("u", color=BLUE)
        v_label = MathTex("v", color=GREEN)
        w_label = MathTex("w=u+v", color=RED)
        
        # Place labels based on critic's specific instructions
        self.place_at_grid(u_label, 'D5', scale_factor=0.7)
        self.place_at_grid(v_label, 'B4', scale_factor=0.7)
        w_label.next_to(w.get_end(), UP+RIGHT, buff=0.1)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.play(Create(u), Write(u_label), Create(v), Write(v_label), Create(w), Write(w_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[0].animate.set_color(WHITE), self.lecture[1].animate.set_color(YELLOW))
        self.play(FadeOut(w), FadeOut(w_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[1].animate.set_color(WHITE), self.lecture[2].animate.set_color(YELLOW))
        self.play(Indicate(u), Indicate(v))
        self.wait(2)
