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
        self.setup_layout("Summary and Conclusion", [
            "Differentiating an integral returns the starting function.", 
            "This provides a closed-loop system for calculus.", 
            "Area calculation is now simplified forever."
        ])
        
        # Formula for animation
        formula = MathTex(r"\frac{d}{dx} \left[ \int_a^x f(t) dt \right] = f(x)", color="#FFFFFF")
        
        # Label
        ft_label = Text("Fundamental Theorem", font_size=24, color="#00CED1")
        # Placeholder asset handling (since none.svg is provided as dummy)
        asset_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg") if False else Dot(color="#00CED1")
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFD700")
        self.place_in_area(formula, 'B3', 'E5', scale_factor=1.0)
        self.play(Write(formula))
        
        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FFD700")
        equals_sign = formula[0][11] # Adjust based on MathTex structure
        self.play(Flash(equals_sign, color="#FFD700"))
        
        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FFD700")
        self.place_in_area(ft_label, 'B1', 'B6', scale_factor=1.0)
        self.play(FadeIn(ft_label, shift=UP))
        self.wait(2)
