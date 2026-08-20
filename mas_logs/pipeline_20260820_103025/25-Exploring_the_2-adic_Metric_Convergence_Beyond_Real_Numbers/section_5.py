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
        self.setup_layout("Synthesis and Takeaway", [
            "Convergence relies on the chosen metric structure.",
            "2-adics reveal hidden arithmetic connections.",
            "Real lines vs. 2-adic trees offer different views."
        ])
        
        # Load Assets
        ruler_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ruler.svg")
        map_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/map.svg")
        globe_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/globe.svg")
        
        # Define visual components
        real_line = NumberLine(x_range=[-3, 3, 1], length=4, color=BLUE)
        real_line.add_numbers(font_size=15)
        real_label = Text("Real Number Line", font_size=18, color=BLUE)
        real_group = VGroup(ruler_icon, real_line, real_label).arrange(DOWN)
        
        tree_node = Dot(color=YELLOW)
        tree_branches = VGroup(Line(ORIGIN, UP*0.8 + LEFT*0.4), Line(ORIGIN, UP*0.8 + RIGHT*0.4)).set_color(YELLOW)
        tree_vis = VGroup(map_icon, tree_node, tree_branches).arrange(DOWN)
        tree_label = Text("2-adic Tree", font_size=18, color=YELLOW)
        tree_group = VGroup(tree_vis, tree_label).arrange(DOWN)
        
        summary_box = RoundedRectangle(corner_radius=0.1, height=1.2, width=3, color=WHITE)
        summary_text = VGroup(
            Text("Distance = Topology", font_size=16, color=WHITE),
            Text("Arithmetic = Structure", font_size=16, color=WHITE)
        ).arrange(DOWN)
        summary_content = VGroup(globe_icon, summary_box, summary_text).arrange(DOWN).scale(0.8)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(BLUE)
        self.place_in_area(real_group, "A4", "B6", scale_factor=0.7)
        self.play(Create(real_group))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        self.place_in_area(tree_group, "C4", "D6", scale_factor=0.7)
        self.play(Create(tree_group))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(GREEN)
        self.place_in_area(summary_content, "E4", "F6", scale_factor=0.6)
        self.play(Create(summary_content))
        self.wait(2)
