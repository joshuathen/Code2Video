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
        lecture_lines = [
            "Total parity points to the secret square.",
            "Flip one coin to force the parity.",
            "Bob reads parity to find the square.",
            "The secret is encoded in the parity.",
            "Success is guaranteed by simple XOR math."
        ]
        self.setup_layout("The Strategy: Parity as a Pointer", lecture_lines)
        
        # Load assets
        coin_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/coin.svg"
        coin1 = SVGMobject(coin_path, color=WHITE)
        coin2 = SVGMobject(coin_path, color=WHITE)

        # Elements
        parity_bits = VGroup(*[Square(side_length=0.5, color=WHITE) for _ in range(4)])
        self.place_at_grid(parity_bits, 'C2', scale_factor=0.65)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#4DA6FF"))
        # Visualize 8x8 grid of bits in #4DA6FF
        grid_visual = VGroup(*[Square(side_length=0.3, color="#4DA6FF", fill_opacity=0.5) for _ in range(16)])
        grid_visual.arrange_in_grid(4, 4, buff=0.1)
        self.place_at_grid(grid_visual, 'B4', scale_factor=0.8)
        self.play(Create(grid_visual))
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FFFF00"))
        # Animate parity calculation as a path through bits in #FFFF00
        path = VMobject(color="#FFFF00").set_points_smoothly([self.grid['B4'], self.grid['C4'], self.grid['C5']])
        self.play(Create(path))
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FF00FF"))
        # Highlight the targeted square by flipping one bit in #FF00FF using a coin
        self.place_at_grid(coin1, 'E5', scale_factor=0.65)
        self.play(FadeIn(coin1))
        
        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#FF8000"))
        # Show parity of the grid matching the target index in #FF8000
        parity_text = Text("Parity Matched", color="#FF8000", font_size=20)
        self.place_at_grid(parity_text, 'D5', scale_factor=1.0)
        self.play(Write(parity_text))
        
        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#00FF00"))
        # Display "Parity Match!" success message in #00FF00 with a coin
        self.place_at_grid(coin2, 'F5', scale_factor=0.65)
        success_text = Text("Success!", color="#00FF00", font_size=24)
        self.place_at_grid(success_text, 'F4', scale_factor=1.0)
        self.play(FadeIn(coin2), Write(success_text))
        self.wait(2)
