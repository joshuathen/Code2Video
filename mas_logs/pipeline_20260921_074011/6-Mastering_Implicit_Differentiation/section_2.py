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
        self.setup_layout("Prerequisite: The Chain Rule Core", [
            "Recall the chain rule for derivatives.",
            "Treat y as a function of x.",
            "Always multiply by dy/dx when differentiating y."
        ])
        
        # === Animation for Lecture Line 1 ===
        chain_rule = MathTex(
            r"\frac{d}{dx} [f(g(x))] = f'(g(x)) \cdot g'(x)",
            font_size=36
        )
        # Apply layout requirement for issue 23, 24, 25, 36
        self.place_in_area(chain_rule, 'B2', 'E4', scale_factor=1.5)
        self.play(Write(chain_rule))
        self.lecture[0].set_color("#FFFFFF")

        # === Animation for Lecture Line 2 ===
        # Use asset reference for g(x)
        # Load asset /scratch/pawsey1357/jthen/Code2Video/assets/icon/box.svg
        box_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/box.svg")
        self.place_at_grid(box_icon, 'D5', scale_factor=0.5)
        
        # Add g(x) text label
        g_x_label = MathTex("y = g(x)", color="#FF00FF")
        g_x_label.next_to(box_icon, UP, buff=0.2)
        
        self.play(FadeIn(box_icon), Write(g_x_label))
        self.lecture[1].set_color("#FF00FF")

        # === Animation for Lecture Line 3 ===
        # Highlight d/dx term for y as dy/dx
        dy_dx_highlight = MathTex(r"\frac{dy}{dx}", color="#00FF00")
        self.place_at_grid(dy_dx_highlight, 'E2', scale_factor=1.2)
        self.play(FadeIn(dy_dx_highlight))
        self.lecture[2].set_color("#00FF00")
        
        self.wait(2)
