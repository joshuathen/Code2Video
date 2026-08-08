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
            "Dual of a dual graph returns the original.",
            "Duality maps original vertices to dual faces.",
            "Duality maps original faces to dual vertices.",
            "Edge count remains the same in duality.",
            "Euler’s characteristic stays constant through duality."
        ]
        self.setup_layout("Interplay: Duality and Euler", lecture_lines)
        
        # Load asset
        grid_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/grid.svg")
        
        # Define basic graph structures
        g_orig = VGroup(
            Dot(color=BLUE), 
            Dot(color=BLUE).shift(RIGHT*1.5), 
            Dot(color=BLUE).shift(UP*1.5),
            Dot(color=BLUE).shift(RIGHT*1.5 + UP*1.5)
        )
        for i in range(len(g_orig)-1):
            g_orig.add(Line(g_orig[i].get_center(), g_orig[i+1].get_center(), color=BLUE_B))
            
        g_dual = VGroup(
            Dot(color=YELLOW),
            Dot(color=YELLOW).shift(RIGHT*0.75 + UP*0.75)
        )
        
        graph_group = VGroup(grid_icon, g_orig, g_dual)
        formula = Text("V - E + F = 2", font_size=24, color=WHITE)
        
        # === Animation for Lecture Line 1 ===
        self.place_in_area(graph_group, 'B2', 'D4', scale_factor=0.75)
        self.play(FadeIn(graph_group))
        self.lecture[0].set_color(BLUE)

        # === Animation for Lecture Line 2 ===
        self.play(Indicate(g_orig), Indicate(g_dual))
        self.lecture[1].set_color(YELLOW)

        # === Animation for Lecture Line 3 ===
        self.play(FadeIn(g_orig.copy()))
        self.lecture[2].set_color(YELLOW)

        # === Animation for Lecture Line 4 ===
        self.play(graph_group.animate.set_opacity(0.5))
        self.lecture[3].set_color(BLUE_C)

        # === Animation for Lecture Line 5 ===
        self.place_at_grid(formula, 'E4', scale_factor=0.7)
        self.play(FadeIn(formula))
        self.lecture[4].set_color(GREEN)
        self.wait(2)
