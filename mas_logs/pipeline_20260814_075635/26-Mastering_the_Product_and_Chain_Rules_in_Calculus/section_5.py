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
        self.setup_layout("Conclusion & Mental Check", [
            "Product rule acts like dancing pairs.",
            "Chain rule is like peeling onions.",
            "Master these for calculus success."
        ])
        
        # Assets
        onion = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/onion.svg")
        dancer = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/dancer.svg")
        
        # === Animation for Lecture Line 1 ===
        # Recap formulas
        product_rule = MathTex(r"(uv)' = u'v + uv'").set_color(WHITE)
        self.place_in_area(product_rule, 'A2', 'B5', scale_factor=1.0)
        self.place_at_grid(dancer, 'A5', scale_factor=0.5)
        self.play(Write(product_rule), FadeIn(dancer))
        self.play(self.lecture[0].animate.set_color(WHITE))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Show mental check flowchart
        chain_rule = MathTex(r"f(g(x))' = f'(g(x)) \cdot g'(x)").set_color("#00FFFF")
        self.place_in_area(chain_rule, 'C2', 'D5', scale_factor=1.0)
        self.place_at_grid(onion, 'D5', scale_factor=0.5)
        self.play(Write(chain_rule), FadeIn(onion))
        self.play(self.lecture[1].animate.set_color("#00FFFF"))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Final summary
        summary = Text("Calculus Success: Practice Daily!", font_size=36, color=WHITE)
        self.place_at_grid(summary, 'E3', scale_factor=1.2)
        self.play(FadeIn(summary))
        self.play(self.lecture[2].animate.set_color(WHITE))
        self.wait(2)
