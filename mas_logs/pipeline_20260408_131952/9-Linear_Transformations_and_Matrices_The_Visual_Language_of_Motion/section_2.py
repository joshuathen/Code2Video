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

class Section2Scene(TeachingScene):
    def construct(self):
        # Initial Setup
        title_text = "Prerequisite: The Power of Basis Vectors"
        # Script alignment: Line text must match teaching content exactly
        lecture_lines = [
            'Meet the basis vectors, i-hat and j-hat, our building blocks.',
            'Scale these arrows to reach any point in space.',
            'We label these unit vectors i-hat and j-hat.'
        ]
        self.setup_layout(title_text, lecture_lines)

        # Configuration
        I_HAT_COLOR = "#FF0000"
        J_HAT_COLOR = "#00FF00"
        GRID_COLOR = "#444444"
        DOT_COLOR = "#FFFFFF"
        LABEL_COLOR = "#FFFFFF"
        HIGHLIGHT_COLOR = "#FFFF00"
        
        # Background Grid (Right Side)
        plane = NumberPlane(
            x_range=[-1, 5, 1],
            y_range=[-1, 4, 1],
            x_length=4.5,
            y_length=4.5,
            background_line_style={"stroke_color": GRID_COLOR, "stroke_opacity": 0.5}
        ).set_z_index(-1)
        self.place_in_area(plane, 'A1', 'F6')
        origin = plane.c2p(0, 0)

        # === Animation for Lecture Line 1 ===
        # "Meet the basis vectors, i-hat and j-hat, our building blocks."
        self.lecture[0].set_color(HIGHLIGHT_COLOR)
        
        i_hat = Arrow(origin, plane.c2p(1, 0), buff=0, color=I_HAT_COLOR, stroke_width=6)
        j_hat = Arrow(origin, plane.c2p(0, 1), buff=0, color=J_HAT_COLOR, stroke_width=6)
        
        self.play(Create(plane), run_time=1.2)
        self.play(GrowArrow(i_hat), GrowArrow(j_hat))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "Scale these arrows to reach any point in space."
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(HIGHLIGHT_COLOR)
        
        target_pt = [3, 2]
        dot = Dot(plane.c2p(*target_pt), color=DOT_COLOR, radius=0.08)
        
        # Show scaling components
        i_scaled = Arrow(origin, plane.c2p(3, 0), buff=0, color=I_HAT_COLOR, stroke_width=4, stroke_opacity=0.7)
        j_scaled = Arrow(plane.c2p(3, 0), plane.c2p(3, 2), buff=0, color=J_HAT_COLOR, stroke_width=4, stroke_opacity=0.7)
        
        self.play(Create(dot))
        # Animate the construction of the point using basis vector scaling
        self.play(ReplacementTransform(i_hat.copy(), i_scaled), run_time=1.2)
        self.play(ReplacementTransform(j_hat.copy(), j_scaled), run_time=1.2)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "We label these unit vectors i-hat and j-hat."
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(HIGHLIGHT_COLOR)
        
        # Position labels using visual anchor system cells requested by VideoCritic
        i_hat_label = Text("i-hat", font_size=20, color=LABEL_COLOR)
        self.place_at_grid(i_hat_label, 'F3', scale_factor=0.6) # Issue 35 fix
        
        j_hat_label = Text("j-hat", font_size=20, color=LABEL_COLOR)
        self.place_at_grid(j_hat_label, 'D1', scale_factor=0.6) # Issue 36 fix (re-applying cell D1 with smaller scale)
        
        point_label = Text("(3, 2)", font_size=20, color=WHITE)
        self.place_at_grid(point_label, 'B6', scale_factor=0.7) # Issue 34 fix
        
        self.play(
            Write(i_hat_label),
            Write(j_hat_label),
            Write(point_label)
        )
        
        # Highlight the original basis vectors with a yellow flash as per description
        self.play(
            Indicate(i_hat, color=HIGHLIGHT_COLOR, scale_factor=1.4),
            Indicate(j_hat, color=HIGHLIGHT_COLOR, scale_factor=1.4),
            run_time=2
        )
        
        self.wait(3)
