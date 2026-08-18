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
        self.setup_layout("Prerequisite: Probability & Visual Mapping", [
            "States are Green, Yellow, Gray.",
            "Imagine a decision tree branching out.",
            "Search space shrinks with every guess."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Draw a probability distribution bar chart using tree asset, color #FF8000
        # Placeholder for tree asset as it's an SVG icon
        tree_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/tree.svg")
        
        bars = VGroup(
            tree_icon.copy(),
            tree_icon.copy(),
            tree_icon.copy(),
            tree_icon.copy()
        ).arrange(RIGHT, aligned_edge=DOWN, buff=0.2)
        
        # Apply color #FF8000 to the bars (icon might need set_color)
        for b in bars:
            b.set_color("#FF8000")
            
        self.place_in_area(bars, 'B4', 'F6', scale_factor=0.6)
        self.play(Create(bars))
        self.lecture[0].set_color("#FF8000")

        # === Animation for Lecture Line 2 ===
        # Highlight the peak of the distribution, color #FF0000
        peak = bars[1]
        self.play(peak.animate.set_color("#FF0000"))
        self.lecture[1].set_color("#FF0000")

        # === Animation for Lecture Line 3 ===
        # Show mapping of states to probabilities, color #00FF00
        mapping_text = VGroup(
            Text("States", font_size=20),
            Text("Mapping", font_size=20)
        ).arrange(DOWN)
        self.place_at_grid(mapping_text, 'E4', scale_factor=0.7)
        self.play(Write(mapping_text))
        mapping_text.set_color("#00FF00")
        self.lecture[2].set_color("#00FF00")
        
        self.wait(2)
