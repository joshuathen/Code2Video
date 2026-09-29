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
            "Wallis formula relates Pi to simple ratios.",
            "The pattern uses even numbers flanking odds.",
            "Each term brings us closer to Pi."
        ]
        self.setup_layout("Deriving the Wallis Formula", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Display ratios 2/1, 2/3, 4/3, 4/5; color numbers #FFFFFF.
        ratio_text = MathTex(r"\frac{2}{1} \cdot \frac{2}{3} \cdot \frac{4}{3} \cdot \frac{4}{5}", color=WHITE)
        self.place_in_area(ratio_text, 'C2', 'C4', scale_factor=0.9)
        self.play(Write(ratio_text))
        self.lecture[0].set_color(YELLOW)

        # === Animation for Lecture Line 2 ===
        # Highlight even numbers in top, odds in bottom; apply color #7FFFD4.
        self.lecture[1].set_color(YELLOW)
        self.wait(1)
        highlight = SurroundingRectangle(ratio_text, color="#7FFFD4", buff=0.1)
        self.play(Create(highlight))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Show full Wallis product sequence expanding; fade in Pi symbol [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/pi.svg] #FF00FF.
        self.lecture[2].set_color(YELLOW)
        
        # Load asset
        pi_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/pi.svg")
        pi_sym = VGroup(MathTex(r"\frac{\pi}{2} = \dots", color="#FF00FF"), pi_icon).arrange(RIGHT)
        
        self.place_in_area(pi_sym, 'E2', 'E4', scale_factor=1.2)
        self.play(FadeIn(pi_sym))
        self.wait(2)
