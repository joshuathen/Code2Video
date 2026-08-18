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
        lecture_lines = [
            "Intermediate Value Theorem: Continuous functions crossing zero change sign.",
            "Imagine a hiker crossing a mountain range.",
            "If the hiker crosses, they must hit zero altitude."
        ]
        self.setup_layout("Prerequisites: The Intermediate Value Theorem (IVT)", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFFFF")
        
        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FFD700")
        
        axes = Axes(x_range=[-1, 5, 1], y_range=[-2, 2, 1], axis_config={"include_tip": False}).scale(0.5)
        func = axes.plot(lambda x: 0.5 * (x - 2), color="#FFD700")
        mountain = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/mountain.svg")
        
        graph_group = VGroup(axes, func, mountain)
        self.place_in_area(graph_group, 'D1', 'F6', scale_factor=0.9)
        self.play(Create(func), FadeIn(mountain), run_time=2)
        
        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#32CD32")
        
        a, b = 1, 3
        dot_a = Dot(axes.c2p(a, func.underlying_function(a)), color="#00FFFF")
        dot_b = Dot(axes.c2p(b, func.underlying_function(b)), color="#00FFFF")
        
        label_a = MathTex("a").next_to(dot_a, DOWN, buff=0.1).scale(0.7)
        label_b = MathTex("b").next_to(dot_b, DOWN, buff=0.1).scale(0.7)
        
        f_a_label = MathTex("f(a) < 0", color="#FF4500").scale(0.6)
        f_b_label = MathTex("f(b) > 0", color="#FF4500").scale(0.6)
        
        self.place_at_grid(f_a_label, 'D2', scale_factor=0.6)
        self.place_at_grid(f_b_label, 'F4', scale_factor=0.6)
        
        self.play(Create(dot_a), Create(dot_b), Write(label_a), Write(label_b), Write(f_a_label), Write(f_b_label))
        
        c = 2
        root = Dot(axes.c2p(c, 0), color="#32CD32")
        hiker = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/hiker.svg")
        hiker.move_to(root.get_center()).scale(0.5)
        
        C_label = MathTex("c", color="#32CD32").scale(0.6)
        self.place_at_grid(C_label, 'E3', scale_factor=0.6)
        
        self.play(FadeIn(root), FadeIn(hiker), Write(C_label))
        self.wait(2)
