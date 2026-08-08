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
        lecture_lines = [
            "Most sphere volume migrates to the shell.",
            "The center becomes hollow as dimensions grow.",
            "Think of an orange with a giant peel.",
            "Almost all mass concentrates near the crust.",
            "Volume shifts away from the inner sphere."
        ]
        self.setup_layout("The 'Empty' Core Paradox", lecture_lines)
        
        # Create visual elements
        # 1. Intuitive Hook
        sphere_core = Circle(radius=1.5, color=WHITE, fill_opacity=0.3)
        self.place_in_area(sphere_core, 'A3', 'B4', scale_factor=0.5)
        
        # 2. Formalization
        integral_text = MathTex(r"V_{shell} = \int_{r-\epsilon}^{r} S_n(r')dr'", color=WHITE)
        self.place_in_area(integral_text, 'C3', 'D4', scale_factor=0.6)
        
        # 3. Visual Synthesis (Shell highlight)
        shell = Annulus(inner_radius=1.0, outer_radius=1.5, color=WHITE, fill_opacity=0.6)
        self.place_at_grid(shell, 'E2', scale_factor=0.5)

        # 4. Analogy (Orange)
        orange_svg = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/orange.svg")
        orange_group = VGroup(orange_svg)
        self.place_at_grid(orange_group, 'E5', scale_factor=0.5)

        # 5. Summary (99%)
        percent_label = Text("99% volume", font_size=32, color=WHITE)
        self.place_at_grid(percent_label, 'B5', scale_factor=0.7)

        # Animations with color changes (Mandatory Constraint)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FF69B4"), sphere_core.animate.set_color("#FF69B4"))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FFD700"), integral_text.animate.set_color("#FFD700"))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#32CD32"), shell.animate.set_color("#32CD32"))

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#FF8C00"), orange_group.animate.set_color("#FF8C00"))

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#FFFFFF"), percent_label.animate.set_color("#FFFFFF"))
        
        self.wait(2)
