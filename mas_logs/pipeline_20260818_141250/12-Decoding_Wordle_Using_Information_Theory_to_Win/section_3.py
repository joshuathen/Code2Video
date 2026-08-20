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
            "Every guess splits possible words.",
            "Feedback categorizes remaining words.",
            "Choose words to split pools equally.",
            "Balanced buckets maximize information.",
            "Information gain reveals answer faster."
        ]
        self.setup_layout("The Core Logic: Information Gain", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Define Information Gain as reduction in entropy.
        self.play(self.lecture[0].animate.set_color(BLUE))
        entropy_text = MathTex(r"Gain(S, A) = H(S) - H(S|A)", font_size=32, color=GREEN)
        self.place_in_area(entropy_text, 'A3', 'A6', scale_factor=0.9)
        self.play(Write(entropy_text))

        # === Animation for Lecture Line 2 ===
        # Feedback categorizes remaining words.
        # Demonstrate how a split reduces entropy on example using bucket.svg
        self.play(self.lecture[0].animate.set_color(WHITE), self.lecture[1].animate.set_color(YELLOW))
        words_pool = VGroup(*[Circle(radius=0.1, color=GRAY, fill_opacity=0.5) for _ in range(12)])
        words_pool.arrange_in_grid(3, 4)
        self.place_in_area(words_pool, 'C3', 'F6', scale_factor=0.8)
        
        bucket_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/bucket.svg", color=BLUE)
        self.place_at_grid(bucket_icon, "D2", scale_factor=0.5)
        self.play(Create(words_pool), FadeIn(bucket_icon))

        # === Animation for Lecture Line 3 ===
        # Choose words to split pools equally.
        # Highlight the "best" attribute for splitting using funnel.svg
        self.play(self.lecture[1].animate.set_color(WHITE), self.lecture[2].animate.set_color(BLUE))
        
        funnel_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/funnel.svg", color=ORANGE)
        self.place_at_grid(funnel_icon, "C2", scale_factor=0.6)
        
        pointer_arrow = Arrow(start=UP, end=DOWN, color=RED).scale(0.5)
        self.place_at_grid(pointer_arrow, 'C2', scale_factor=1.0)
        
        self.play(FadeIn(funnel_icon), GrowArrow(pointer_arrow))

        # === Animation for Lecture Line 4 ===
        # Balanced buckets maximize information.
        self.play(self.lecture[2].animate.set_color(WHITE), self.lecture[3].animate.set_color(GREEN))
        bucket1 = Rectangle(height=1.0, width=0.8, color=BLUE).move_to(self.grid["F2"])
        bucket2 = Rectangle(height=1.0, width=0.8, color=BLUE).move_to(self.grid["F4"])
        self.play(Create(bucket1), Create(bucket2))

        # === Animation for Lecture Line 5 ===
        # Information gain reveals answer faster.
        self.play(self.lecture[3].animate.set_color(WHITE), self.lecture[4].animate.set_color(YELLOW))
        check_mark = Tex(r"$\checkmark$", color=RED).scale(2).move_to(self.grid["E3"])
        self.play(Write(check_mark))
        self.wait(2)
