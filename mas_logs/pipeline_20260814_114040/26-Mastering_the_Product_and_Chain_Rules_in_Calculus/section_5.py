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
        lecture_lines = [
            "Solve problems needing both Product and Chain Rules.",
            "Identify the primary rule skeleton first.",
            "Use Chain Rule for inner components.",
            "Differentiate outside while keeping inside fixed.",
            "Finally multiply by the inner derivative."
        ]
        self.setup_layout("Synthesis & Summary", lecture_lines)
        
        # Colors
        product_color = "#3498db"
        chain_color = "#e67e22"
        skeleton_color = "#F39C12"
        
        # Product Rule Skeleton: f(x)g(x) -> f'g + fg'
        product_eq = MathTex(r"d(f \cdot g) = f' \cdot g + f \cdot g'", color=product_color)
        # Chain Rule: f(g(x)) -> f'(g(x)) \cdot g'(x)
        chain_eq = MathTex(r"d(f(g(x))) = f'(g(x)) \cdot g'(x)", color=chain_color)
        
        # Assets
        puzzle_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/puzzle.svg").scale(0.3)
        skeleton_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/skeleton.svg").scale(0.3).set_color(skeleton_color)
        gears_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/gears.svg").scale(0.3)

        # Applying fixes from critique
        self.place_in_area(product_eq, 'A2', 'B5', scale_factor=0.9)
        self.place_in_area(chain_eq, 'C2', 'D5', scale_factor=0.9)
        
        # === Animation for Lecture Line 1 ===
        self.place_at_grid(puzzle_icon, 'A6')
        self.play(FadeIn(product_eq), FadeIn(chain_eq), FadeIn(puzzle_icon))
        self.play(self.lecture[0].animate.set_color(WHITE))
        
        # === Animation for Lecture Line 2 ===
        self.place_at_grid(skeleton_icon, 'A5')
        self.play(Indicate(product_eq), FadeIn(skeleton_icon))
        self.play(self.lecture[1].animate.set_color(skeleton_color))
        
        # === Animation for Lecture Line 3 ===
        self.play(Indicate(chain_eq))
        self.play(self.lecture[2].animate.set_color(chain_color))
        
        # === Animation for Lecture Line 4 ===
        outside_part = MathTex(r"f'(g(x))", color=WHITE)
        inner_part = MathTex(r"\cdot g'(x)", color=WHITE)
        components_group = VGroup(outside_part, inner_part).arrange(RIGHT)
        self.place_in_area(components_group, 'D4', 'D6', scale_factor=0.8)
        
        self.play(Write(components_group))
        self.play(self.lecture[3].animate.set_color(WHITE))
        
        # === Animation for Lecture Line 5 ===
        self.place_at_grid(gears_icon, 'F6')
        self.play(Flash(gears_icon, color=WHITE), FadeIn(gears_icon))
        self.play(self.lecture[4].animate.set_color(WHITE))
        
        self.wait(2)
