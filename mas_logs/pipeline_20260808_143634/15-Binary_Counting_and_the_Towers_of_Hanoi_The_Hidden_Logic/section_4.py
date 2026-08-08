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
            "Binary counter increments steadily.",
            "Bits match disk movements.",
            "Algorithm automates puzzle solving.",
            "Visualizing the pattern clearly.",
            "Counting solves the towers."
        ]
        self.setup_layout("Visualizing the Algorithm", lecture_lines)
        
        # Colors for lines
        colors = ["#FFD700", "#00BFFF", "#32CD32", "#FF4500", "#9370DB"]
        
        # Load assets
        tower_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/tower.svg")
        disk_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/disk.svg")
        
        # Apply critic requested positioning/objects
        # Tree system representation
        tree_system = VGroup(tower_icon.copy(), disk_icon.copy())
        self.place_in_area(tree_system, 'A3', 'F6', scale_factor=0.6)
        
        # Grid labels (placeholder for demo)
        grid_labels = Text("Grid", font_size=20)
        self.place_at_grid(grid_labels, 'B5', scale_factor=0.75)
        
        # General grid system placeholder
        grid_system = Rectangle(width=2, height=2, color=BLUE)
        self.place_in_area(grid_system, 'C2', 'E5', scale_factor=0.5)

        self.add(tree_system, grid_labels, grid_system)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(colors[0]))
        self.wait(0.5)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(colors[1]))
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(colors[2]))
        self.wait(0.5)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(colors[3]))
        # Fade out a completed subtree in gray
        self.play(tree_system.animate.set_color("#808080"))
        self.wait(0.5)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(colors[4]))
        self.play(FadeIn(disk_icon.copy().shift(RIGHT*2)))
        self.wait(1)
