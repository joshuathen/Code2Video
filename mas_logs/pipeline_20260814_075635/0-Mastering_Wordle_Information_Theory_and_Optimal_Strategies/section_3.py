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
            "High frequency letters are superior.",
            "Minimax minimizes the maximum remaining words.",
            "Max entropy maximizes information gained.",
            "Compare guess entropy for optimization.",
            "Visualize remaining sets with charts."
        ]
        self.setup_layout("Algorithmic Strategy: The 'Best' First Guess", lecture_lines)
        
        # Create objects for animation
        # Asset Loading: SVGMobject is used for assets
        computer_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/computer.svg")
        keyboard_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/keyboard.svg")
        
        candidate_word = VGroup(
            Text("CRANE", font_size=36),
            computer_icon
        ).arrange(DOWN)
        
        split_visual = VGroup(
            Rectangle(width=1, height=2, color=BLUE),
            Rectangle(width=1, height=1, color=BLUE),
            Rectangle(width=1, height=0.5, color=BLUE)
        ).arrange(RIGHT, buff=0.2)
        
        # === Animation for Lecture Line 1 ===
        # Fix overlapping: self.place_at_grid(CRANE, 'A4', scale_factor=1.0)
        self.play(FadeIn(self.place_at_grid(candidate_word, 'A4', 0.6)))
        self.play(self.lecture[0].animate.set_color("#FFFF00"))

        # === Animation for Lecture Line 2 ===
        # Fix clutter: self.place_in_area(BoxGroup, 'D2', 'F5', scale_factor=0.6)
        self.play(FadeIn(self.place_in_area(split_visual, 'D2', 'F5', 0.6)))
        self.play(self.lecture[1].animate.set_color("#00FFFF"))

        # === Animation for Lecture Line 3 ===
        # Add keyboard icon for entropy
        keyboard_icon.scale(0.5).next_to(split_visual, RIGHT)
        self.add(keyboard_icon)
        
        # Fix descriptive text positioning: self.place_in_area(SVM_Description, 'B1', 'C3', scale_factor=0.7)
        # Assuming SVG or text label for the grid description
        description = Text("Entropy Branch", font_size=24)
        self.place_in_area(description, 'B1', 'C3', 0.7)
        self.play(FadeIn(description))
        
        self.play(self.lecture[2].animate.set_color("#FF00FF"))

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(GREEN))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(YELLOW))
        self.wait(2)
