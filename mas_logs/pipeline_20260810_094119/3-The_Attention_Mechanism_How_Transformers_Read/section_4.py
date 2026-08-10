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
        self.setup_layout("Softmax and Weighting", [
            "Scores can be large and unmanageable.",
            "Softmax maps scores to a zero-one range.",
            "Attention becomes a probability distribution.",
            "The spotlight highlights relevant words clearly.",
            "Weights sum up to one perfectly."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Scores [2, 1, -1]
        scores = VGroup(*[MathTex(s, color=WHITE) for s in ["2", "1", "-1"]])
        scores.arrange(DOWN)
        self.place_at_grid(scores, 'B3', scale_factor=0.8)
        
        # Integration of Asset
        spotlight_img = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/spotlight.svg")
        self.place_at_grid(spotlight_img, 'B2', scale_factor=0.5)
        
        self.play(FadeIn(scores), FadeIn(spotlight_img))
        self.play(self.lecture[0].animate.set_color(WHITE))

        # === Animation for Lecture Line 2 ===
        # Transition to Softmax [0.67, 0.24, 0.09]
        probs = VGroup(*[MathTex(s, color="#FF5733") for s in ["0.67", "0.24", "0.09"]])
        probs.arrange(DOWN)
        self.place_at_grid(probs, 'B4', scale_factor=0.8)
        self.play(Transform(scores, probs))
        self.play(self.lecture[1].animate.set_color("#FF5733"))

        # === Animation for Lecture Line 3 ===
        dist_label = Text("Attention Weights", font_size=20, color="#33FF57")
        self.place_at_grid(dist_label, 'D3', scale_factor=1.0)
        self.play(Write(dist_label))
        self.play(self.lecture[2].animate.set_color("#33FF57"))

        # === Animation for Lecture Line 4 ===
        # Using the spotlight asset as requested for the weighted highlight
        spotlight = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/spotlight.svg")
        self.place_at_grid(spotlight, 'E3', scale_factor=0.7)
        self.play(FadeIn(spotlight))
        self.play(spotlight.animate.shift(RIGHT*1.5))
        self.play(self.lecture[3].animate.set_color("#33FF57"))

        # === Animation for Lecture Line 5 ===
        # Sum = 1
        sum_text = MathTex(r"\\sum = 1", color="#33FF57")
        self.place_at_grid(sum_text, 'E5', scale_factor=0.8)
        self.play(Write(sum_text))
        self.play(self.lecture[4].animate.set_color("#33FF57"))
        self.wait(2)
