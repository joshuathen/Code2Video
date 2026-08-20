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
        lecture_lines = [
            "Calculate entropy for all words.",
            "Select the highest scoring guess.",
            "Filter dictionary based on feedback.",
            "Repeat until the word is found.",
            "Watch entropy plummet each turn."
        ]
        self.setup_layout("Application: The Strategy in Action", lecture_lines)
        
        # --- Create Animation Objects ---
        # Represent dataset (e.g., small squares) with asset reference
        dataset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/dictionary.svg")
        # Visual Critic Issue 30: Adjust dataset positioning
        self.place_in_area(dataset, 'A4', 'C6', scale_factor=0.6)
        
        # Selection highlight
        highlight = Rectangle(width=0.4, height=0.4, color=YELLOW, stroke_width=4)
        
        # Subset visual
        subset_group = VGroup()

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(dataset))
        self.lecture[0].set_color(YELLOW)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE), 
            self.lecture[1].animate.set_color(YELLOW)
        )
        # Visual Critic Issue 31: Adjust highlight positioning
        self.place_at_grid(highlight, 'B5', scale_factor=0.8)
        self.play(Create(highlight))

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE), 
            self.lecture[2].animate.set_color(YELLOW)
        )
        # Assuming subset objects based on original logic
        subset1 = Square(side_length=0.3, fill_opacity=0.5, color=GREEN).move_to(self.grid["E1"])
        subset2 = Square(side_length=0.3, fill_opacity=0.5, color=RED).move_to(self.grid["E4"])
        self.play(
            FadeOut(highlight),
            FadeIn(subset1),
            FadeIn(subset2)
        )

        # === Animation for Lecture Line 4 ===
        self.play(
            self.lecture[2].animate.set_color(WHITE), 
            self.lecture[3].animate.set_color(YELLOW)
        )
        self.play(FadeOut(dataset), FadeOut(subset1), FadeOut(subset2))
        
        small_dataset = VGroup(*[Square(side_length=0.3, fill_opacity=0.5, color=BLUE) for _ in range(4)])
        small_dataset.arrange(RIGHT, buff=0.2)
        self.place_at_grid(small_dataset, "B3")
        self.play(FadeIn(small_dataset))

        # === Animation for Lecture Line 5 ===
        self.play(
            self.lecture[3].animate.set_color(WHITE), 
            self.lecture[4].animate.set_color(YELLOW)
        )
        entropy_text = Text("Entropy: High -> Low", font_size=24, color=RED)
        # Visual Critic Issue 32: Adjust entropy_text positioning
        self.place_at_grid(entropy_text, 'E5', scale_factor=1.0)
        self.play(Write(entropy_text))
        self.wait(2)
