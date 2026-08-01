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
        # Setup layout with title and lecture lines
        self.setup_layout(
            "Summary and Real-World Application: Torque", 
            [
                "Geometry and transformations unite in the cross product.", 
                "In physics, torque is the cross product of r and F.", 
                "These principles govern rotation from motors to planets."
            ]
        )
        
        # Define colors
        COLOR_R = "#58C4DD"  # Cyan
        COLOR_F = "#83C167"  # Green
        COLOR_TAU = "#F8B195" # Peach
        
        # === Animation for Lecture Line 1 ===
        # Visualization: Bolt and Wrench [Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/wrench.svg] 
        # as vector r and Force vector F
        self.play(self.lecture[0].animate.set_color(COLOR_R))
        
        bolt = VGroup(
            Dot(self.grid['D3'], color=GRAY, radius=0.15),
            RegularPolygon(n=6, color=WHITE, stroke_width=2).scale(0.2).move_to(self.grid['D3'])
        )
        
        # Wrench represents vector r (position vector)
        wrench = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/wrench.svg")
        wrench.set_color(COLOR_R)
        # Position wrench between bolt (D3) and handle end (D5)
        self.place_in_area(wrench, 'D3', 'D5', scale_factor=0.8)
        
        # Label for r
        r_label = Text("r", color=COLOR_R, slant=ITALIC)
        self.place_at_grid(r_label, 'E4', scale_factor=0.8)
        
        # Vector F (Force applied at the handle end of the wrench)
        f_vec = Arrow(start=self.grid['D5'], end=self.grid['B5'], color=COLOR_F, buff=0)
        f_label = Text("F", color=COLOR_F, slant=ITALIC)
        self.place_at_grid(f_label, 'C6', scale_factor=0.8)
        
        self.play(
            Create(bolt),
            FadeIn(wrench),
            Write(r_label),
            run_time=1.5
        )
        self.play(
            GrowArrow(f_vec),
            Write(f_label),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Visualization: Torque vector tau appearing at the bolt, perpendicular to r and F
        self.play(self.lecture[1].animate.set_color(COLOR_TAU))
        
        # tau_vec is perpendicular to r and F (pointing up)
        tau_vec = Arrow(start=self.grid['D3'], end=self.grid['B3'], color=COLOR_TAU, buff=0)
        tau_label = Text("τ = r × F", color=COLOR_TAU, slant=ITALIC)
        
        # Resolve Issue 35/37: Position tau formula to avoid overlapping vector tip
        self.place_in_area(tau_label, 'A3', 'B5', scale_factor=0.7)
        
        self.play(
            GrowArrow(tau_vec),
            Write(tau_label),
            run_time=1.5
        )
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # Visualization: Final message synthesized
        self.play(self.lecture[2].animate.set_color(WHITE))
        
        # Fade out elements to clear space
        self.play(
            FadeOut(bolt, wrench, r_label, f_vec, f_label, tau_vec, tau_label),
            run_time=1
        )
        
        final_text = Text("Geometry + Algebra\n=\nCross Product", color=WHITE, t2c={"Cross Product": COLOR_TAU})
        # Resolve Issue 36: Position text to avoid crowded center-screen layout
        self.place_in_area(final_text, 'B3', 'E6', scale_factor=0.8)
        
        self.play(Write(final_text), run_time=2)
        self.wait(3)
