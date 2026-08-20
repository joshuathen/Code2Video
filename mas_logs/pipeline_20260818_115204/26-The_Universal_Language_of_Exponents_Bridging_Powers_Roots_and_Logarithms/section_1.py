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
        lecture_lines = ["Start with a base and exponent.", "Visualize growth as a tree.", "Example: 2 raised to 3 is 8."]
        self.setup_layout("Prerequisite Warm-up: The Exponential Foundation", lecture_lines)
        
        # Paths for assets
        tree_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/tree.svg"
        
        # === Animation for Lecture Line 1 ===
        # Show label 'Base: b' in #FFFFFF at left, exponent 'x' in #FFCC00
        # Incorporate tree asset as per instruction 16
        tree1 = SVGMobject(tree_path)
        self.place_at_grid(tree1, "A2", scale_factor=0.3)
        self.add(tree1)

        base_label = Text("Base: b", color=WHITE)
        self.place_at_grid(base_label, "B1", scale_factor=0.8) # Fix: Issue 22
        self.play(FadeIn(base_label))
        self.lecture[0].set_color(YELLOW)

        exp_label = Text("Exponent: x", color="#FFCC00")
        self.place_at_grid(exp_label, "B5", scale_factor=0.8) # Fix: Issue 22
        self.play(FadeIn(exp_label))

        # === Animation for Lecture Line 2 ===
        # Visualize growth as a tree
        tree = SVGMobject(tree_path)
        self.place_at_grid(tree, "C3", scale_factor=0.8) # Fix: Issue 20
        self.play(DrawBorderThenFill(tree))
        self.lecture[1].set_color(GREEN)

        # === Animation for Lecture Line 3 ===
        # Example: 2 raised to 3 is 8.
        # Animate result 'y' in #00FFCC appearing to the right.
        eq = MathTex("2^3 = 8", color=WHITE)
        self.place_in_area(eq, "D2", "E4", scale_factor=1.2) # Fix: Issue 21
        self.play(Write(eq))
        
        y_label = Text("Result: y", color="#00FFCC")
        self.place_at_grid(y_label, "E6", scale_factor=0.8) # Fix: Issue 22
        self.play(FadeIn(y_label))
        
        # Add tree asset for line 3
        tree2 = SVGMobject(tree_path)
        self.place_at_grid(tree2, "F4", scale_factor=0.3)
        self.play(FadeIn(tree2))
        
        self.lecture[2].set_color("#00FFCC")
        self.play(Indicate(eq))
        self.wait(1)
