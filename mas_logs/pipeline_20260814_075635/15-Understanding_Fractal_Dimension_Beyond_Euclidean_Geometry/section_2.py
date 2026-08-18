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
        self.setup_layout("Prerequisite: The Scaling Rule", [
            "Scaling a shape by factor r increases measure.",
            "The new measure grows by factor N.",
            "We define dimension D using N = r^D."
        ])
        
        # Asset path
        asset_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/square.svg"

        # 1. Introduce square [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/square.svg] (side length 1, color #FFFFFF).
        # Fix 24, 26: Use place_in_area, 'A3', 'C6', scale 0.9
        square = SVGMobject(asset_path, color=WHITE).set_fill(WHITE, opacity=0.3)
        self.place_in_area(square, 'A3', 'C6', scale_factor=0.9)
        self.play(FadeIn(square))
        self.lecture[0].set_color("#33CCFF")
        self.wait(1)

        # 2. Scale side by 3, resulting in 9 squares [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/square.svg] (color #33FF57).
        # Fix 24, 26: Use place_in_area, 'A3', 'C6', scale 0.9
        grid_sq = VGroup(*[SVGMobject(asset_path, color="#33FF57").set_fill("#33FF57", opacity=0.3) for _ in range(9)])
        grid_sq.arrange_in_grid(3, 3, buff=0)
        self.place_in_area(grid_sq, 'A3', 'C6', scale_factor=0.9)
        self.play(ReplacementTransform(square, grid_sq))
        self.lecture[1].set_color("#33FF57")
        self.wait(1)

        # 3. Show N=9, S=3 in D = log(N)/log(S).
        formula = MathTex(r"N = r^D \Rightarrow D = \frac{\log(N)}{\log(r)}", color=YELLOW)
        params = MathTex(r"N = 9, r = 3", color=WHITE)
        VGroup(formula, params).arrange(DOWN)
        self.place_at_grid(VGroup(formula, params), 'E3')
        self.play(Write(formula), Write(params))
        self.lecture[2].set_color(YELLOW)
        self.wait(1)

        # 4. Calculate log(9)/log(3) to yield 2.
        # Fix 25: Use place_in_area, 'D3', 'E5', scale 0.8
        result = MathTex(r"D = \frac{\log(9)}{\log(3)} = 2", color=YELLOW)
        self.place_in_area(result, 'D3', 'E5', scale_factor=0.8)
        self.play(ReplacementTransform(formula, result))
        self.wait(1)

        # 5. Conclude D = 2 (Euclidean dimension) [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/square.svg].
        # Fix 25: Use place_at_grid, 'F4', scale 0.7
        conclusion_icon = SVGMobject(asset_path, color="#FF33CC").scale(0.3)
        conclusion_text = Text("D = 2 (Euclidean)", color="#FF33CC", font_size=24)
        conclusion = VGroup(conclusion_icon, conclusion_text).arrange(RIGHT)
        self.place_at_grid(conclusion, 'F4', scale_factor=0.7)
        self.play(Write(conclusion))
        self.wait(2)
