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
        lecture_lines_text = ["Roots are inverse growth operations.", "We shrink the tree to seeds.", "If 2^6 is 64.", "The 6th root of 64 is 2.", "Roots help find the base."]
        self.setup_layout("The Inverse Operation: Roots", lecture_lines_text)
        
        # === Animation for Lecture Line 1 ===
        # Show a square root symbol √ in #FFD700 (Gold).
        root_sym = MathTex(r"\sqrt{\cdot}", color="#FFD700")
        self.place_at_grid(root_sym, 'B2', scale_factor=3)
        self.play(FadeIn(root_sym))
        self.lecture[0].set_color("#FFD700")

        # === Animation for Lecture Line 2 ===
        # Shrink the tree diagram from Section 1 to a seed [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/seed.svg], #87CEEB (SkyBlue).
        seed = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/seed.svg", color="#87CEEB")
        self.place_at_grid(seed, 'B5', scale_factor=0.5)
        self.play(FadeIn(seed))
        self.lecture[1].set_color("#87CEEB")

        # === Animation for Lecture Line 3 ===
        # Display '2^6 = 64' then highlight 64, #FF4500 (OrangeRed).
        eq = MathTex("2^6 = 64")
        self.place_at_grid(eq, 'D3', scale_factor=1.5)
        self.play(Write(eq))
        highlight = SurroundingRectangle(eq[0][-2:], color="#FF4500")
        self.play(Create(highlight))
        self.lecture[2].set_color("#FF4500")

        # === Animation for Lecture Line 4 ===
        # Show '6th root of 64 = 2' in #32CD32 (LimeGreen).
        root_eq = MathTex(r"\sqrt[6]{64} = 2", color="#32CD32")
        self.place_at_grid(root_eq, 'E3', scale_factor=1.5)
        self.play(Write(root_eq))
        self.lecture[3].set_color("#32CD32")

        # === Animation for Lecture Line 5 ===
        # Draw an arrow from 64 back to 2, #FFFFFF (White).
        arrow = Arrow(start=eq[0][-2:].get_center(), end=root_eq[0][-1].get_center(), color=WHITE)
        self.play(Create(arrow))
        self.lecture[4].set_color(WHITE)
        self.wait(2)
