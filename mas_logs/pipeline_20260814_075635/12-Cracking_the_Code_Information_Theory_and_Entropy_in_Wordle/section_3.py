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
        self.setup_layout("The Strategy: Maximizing Information Gain", [
            "Wordle strategy maximizes information gain.",
            "Choose words that shrink pools.",
            "Information gain reduces the decision tree."
        ])
        
        # Elements
        keyboard_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/keyboard.svg", color=WHITE)
        root = Circle(radius=0.2, color=WHITE, fill_opacity=1)
        
        # Apply specific requested positioning
        self.place_at_grid(root, 'C4', scale_factor=0.7)
        self.place_at_grid(keyboard_icon, 'B4', scale_factor=0.5)
        
        # Tree construction
        tree_structure = VGroup(root)
        layer1 = [Circle(radius=0.15, color=WHITE, fill_opacity=1) for _ in range(3)]
        for i, node in enumerate(layer1):
            # Placing nodes in specific areas
            self.place_at_grid(node, f'D{i+2}')
            line = Line(root.get_center(), node.get_center(), color="#00FF00")
            tree_structure.add(line, node)
        
        self.place_in_area(tree_structure, 'C3', 'E5', scale_factor=0.6)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFFFF")
        self.play(Create(tree_structure), FadeIn(keyboard_icon))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#00FF00")
        highlight = Circle(radius=0.3, color="#00FF00", fill_opacity=0).move_to(tree_structure[2].get_center())
        self.play(Create(highlight))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FF0000")
        to_prune = VGroup(tree_structure[3], tree_structure[4], tree_structure[5], tree_structure[6])
        self.play(FadeOut(to_prune), FadeOut(tree_structure[1]))
        
        # Move icon for final prompt
        self.play(keyboard_icon.animate.move_to(self.grid['E4']))
        self.wait(1)
