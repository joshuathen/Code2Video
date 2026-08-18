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

class Section5Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Summary & Complexity Analysis", [
            "Total moves follow the two-to-n pattern.", 
            "Binary math simplifies this complex puzzle.", 
            "Exponential growth at work in ten disks. [Asset: growth_graph]"
        ])
        self.lecture.set_opacity(1)

        # Asset loading
        disk_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/disk.svg")

        # === Animation for Lecture Line 1 ===
        # Display 2^n - 1 formula
        formula = MathTex(r"2^n - 1", color=YELLOW)
        # Apply fix for Issue 34/49
        self.place_at_grid(formula, 'B3', scale_factor=1.2)
        # Integrate disk icon
        disk1 = disk_icon.copy().scale(0.3)
        self.place_at_grid(disk1, 'B2')
        self.play(Write(formula), FadeIn(disk1))
        self.lecture[0].set_color(YELLOW)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Summary of complexity O(2^n)
        complexity = MathTex(r"O(2^n)", color=BLUE)
        # Apply fix for Issue 35/50
        self.place_at_grid(complexity, 'D3', scale_factor=1.2)
        # Integrate disk icon
        disk2 = disk_icon.copy().scale(0.3)
        self.place_at_grid(disk2, 'D2')
        self.play(FadeIn(complexity), FadeIn(disk2))
        self.lecture[1].set_color(BLUE)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Exponential growth graph (Asset: growth_graph)
        graph = Axes(x_range=[0, 10, 2], y_range=[0, 1000, 200], axis_config={"include_tip": True})
        curve = graph.plot(lambda x: 2**x, color=GREEN)
        graph_group = VGroup(graph, curve)
        # Apply fix for Issue 33/48
        self.place_in_area(graph_group, 'C4', 'F6', scale_factor=0.4)
        self.play(Create(graph_group))
        self.lecture[2].set_color(GREEN)
        self.wait(2)
