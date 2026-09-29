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
        self.setup_layout("Strategy 2: The Pruning Technique", 
                          ["Refine strategy after initial feedback.", 
                           "Switch to candidate reduction logic.", 
                           "Balance letter identification with positional constraints."])
        
        # === Animation for Lecture Line 1 ===
        # Visualize tree structure - CRANE root
        root = Circle(radius=0.3, color=BLUE, fill_opacity=0.5)
        label1 = Text("CRANE", font_size=16)
        tree_group = VGroup(root, label1)
        self.place_at_grid(tree_group, 'C5', scale_factor=0.9)
        self.play(Create(tree_group))
        self.lecture[0].set_color(BLUE)

        # === Animation for Lecture Line 2 ===
        # Branches representing possible worlds (10 vs 80)
        # Using shears asset
        shears = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/shears.svg", color=WHITE)
        
        branch_optimal = VGroup(*[Circle(radius=0.15, color=GREEN, fill_opacity=0.5) for _ in range(3)])
        branch_optimal.arrange(DOWN, buff=0.2)
        self.place_in_area(branch_optimal, 'B6', 'F6', scale_factor=0.75)
        
        self.play(Create(branch_optimal), FadeIn(shears.next_to(branch_optimal, LEFT)))
        self.lecture[1].set_color(GREEN)

        # === Animation for Lecture Line 3 ===
        # Zoom in/focus on remaining valid word list
        final_list = Text("Valid List", font_size=20)
        self.place_at_grid(final_list, 'C6', scale_factor=0.8)
        self.play(FadeOut(shears), FadeIn(final_list))
        self.lecture[2].set_color(YELLOW)
        self.wait(2)
