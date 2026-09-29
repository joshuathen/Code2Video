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

class Section1Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Introduction: The Uncertainty Problem", [
            "Wordle is essentially a challenging search problem.",
            "Uncertainty arises when all words seem equally probable.",
            "We use decision trees to narrow the search.",
            "Each guess helps us halve the candidate pool.",
            "Information theory guides us to the best guess."
        ])
        
        # Load asset
        wordle_icon_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/wordle.svg"
        wordle_icon = SVGMobject(wordle_icon_path)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFFFF")
        title_icon = wordle_icon.copy()
        self.place_at_grid(title_icon, "A5", scale_factor=0.3)
        self.play(FadeIn(self.title), FadeIn(title_icon))
        
        # Show a grid representing 2,315 possible Wordle words (#00FF00)
        word_grid = VGroup(*[SVGMobject(wordle_icon_path, color="#00FF00") for _ in range(25)])
        word_grid.arrange_in_grid(rows=5, cols=5, buff=0.1)
        # Using mandated fix from Issue 21/36
        self.place_in_area(word_grid, 'A3', 'C5', scale_factor=0.6)
        self.play(Create(word_grid))

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(GRAY)
        self.lecture[1].set_color("#00FF00")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(GRAY)
        self.lecture[2].set_color("#FF00FF")
        # Add a branching structure
        tree = VGroup(Line(ORIGIN, UP*0.5), Line(ORIGIN, DOWN*0.5)).rotate(PI/2)
        self.place_at_grid(tree, "B4", scale_factor=0.5)
        self.play(Create(tree))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(GRAY)
        self.lecture[3].set_color("#00FFFF")
        # Halving animation
        self.play(word_grid.animate.set_color("#00FFFF"), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(GRAY)
        self.lecture[4].set_color("#FFFF00")
        # Highlight path
        path_icon = wordle_icon.copy()
        path_icon.set_color("#FFFF00")
        self.place_at_grid(path_icon, "D3", scale_factor=0.5)
        # Add labels per suggestions
        target_word_label = Text("Target Word", font_size=20, color="#FF0000")
        self.place_at_grid(target_word_label, 'D4', scale_factor=0.7)
        self.play(FadeIn(path_icon), FadeIn(target_word_label))
        self.wait(2)
