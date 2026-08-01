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
        self.setup_layout(
            "Why it Matters: The Power of Generalization",
            [
                "Abstract proofs apply to arrows, functions, and matrices.",
                "One theorem can solve problems across many fields.",
                "This generalization is the core power of linear algebra."
            ]
        )

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        # Physics Icon (Atom) - Issue 17: Use Asset
        # Issue 29: Move to A2, scale 0.6
        physics_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/atom.svg").set_color(WHITE)
        self.place_at_grid(physics_icon, "A2", scale_factor=0.6)
        physics_label = Text("Physics", font_size=16).next_to(physics_icon, DOWN, buff=0.1)
        
        # Economics Icon (Graph)
        # Issue 29: Move to C2, scale 0.6
        axes = Axes(x_range=[0, 3, 1], y_range=[0, 3, 1], x_length=0.8, y_length=0.8, 
                    axis_config={"include_ticks": False, "stroke_width": 2}).set_color(WHITE)
        graph_line = VMobject(color=WHITE, stroke_width=3)
        graph_line.set_points_as_corners([
            axes.c2p(0, 0.5, 0),
            axes.c2p(1, 1.2, 0),
            axes.c2p(2, 1.0, 0),
            axes.c2p(3, 2.5, 0)
        ])
        economics_icon = VGroup(axes, graph_line)
        self.place_at_grid(economics_icon, "C2", scale_factor=0.6)
        economics_label = Text("Economics", font_size=16).next_to(economics_icon, DOWN, buff=0.1)

        # AI Icon (Nodes)
        # Issue 29: Move to E2, scale 0.6
        n1_pos = ORIGIN
        n2_pos = RIGHT*0.6 + UP*0.4
        n3_pos = RIGHT*0.6 + DOWN*0.4
        n4_pos = RIGHT*1.2
        n1 = Dot(radius=0.08, color=WHITE, point=n1_pos)
        n2 = Dot(radius=0.08, color=WHITE, point=n2_pos)
        n3 = Dot(radius=0.08, color=WHITE, point=n3_pos)
        n4 = Dot(radius=0.08, color=WHITE, point=n4_pos)
        l1 = Line(n1_pos, n2_pos, stroke_width=2, color=WHITE)
        l2 = Line(n1_pos, n3_pos, stroke_width=2, color=WHITE)
        l3 = Line(n2_pos, n4_pos, stroke_width=2, color=WHITE)
        l4 = Line(n3_pos, n4_pos, stroke_width=2, color=WHITE)
        ai_icon = VGroup(n1, n2, n3, n4, l1, l2, l3, l4)
        self.place_at_grid(ai_icon, "E2", scale_factor=0.6)
        ai_label = Text("AI", font_size=16).next_to(ai_icon, DOWN, buff=0.1)

        self.play(
            FadeIn(physics_icon), Write(physics_label),
            FadeIn(economics_icon), Write(economics_label),
            FadeIn(ai_icon), Write(ai_label)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)

        # Master Key
        key_handle = Circle(radius=0.3, color="#FFD700", stroke_width=4)
        key_shaft = Rectangle(width=1.0, height=0.1, color="#FFD700", fill_opacity=1).next_to(key_handle, RIGHT, buff=0)
        key_tooth1 = Rectangle(width=0.1, height=0.2, color="#FFD700", fill_opacity=1).move_to(key_shaft.get_right() + LEFT*0.1 + DOWN*0.1)
        key_tooth2 = Rectangle(width=0.1, height=0.15, color="#FFD700", fill_opacity=1).move_to(key_shaft.get_right() + LEFT*0.3 + DOWN*0.075)
        key_label_text = Text("Linear Algebra", font_size=14, color="#FFD700").next_to(key_shaft, UP, buff=0.1)
        master_key = VGroup(key_handle, key_shaft, key_tooth1, key_tooth2, key_label_text)
        
        # Issue 30: Place in area C4 to E6, scale 0.8
        self.place_in_area(master_key, "C4", "E6", scale_factor=0.8)
        
        self.play(FadeIn(master_key, shift=LEFT))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)

        # Key rotation to simulate unlocking
        self.play(Rotate(master_key, angle=30*DEGREES, about_point=key_handle.get_center(), rate_func=wiggle))
        
        # Success state: Icons turn green
        self.play(
            physics_icon.animate.set_color("#00FF00"),
            physics_label.animate.set_color("#00FF00"),
            economics_icon.animate.set_color("#00FF00"),
            economics_label.animate.set_color("#00FF00"),
            ai_icon.animate.set_color("#00FF00"),
            ai_label.animate.set_color("#00FF00"),
            run_time=1.5
        )
        
        self.wait(2)
        self.lecture[2].set_color(WHITE)
