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
        # Setup title and lecture lines
        title_text = "Prerequisite Review: What is a Basis?"
        lecture_lines = [
            "Standard unit vectors define our common grid.",
            "A basis acts as a set of measuring rulers.",
            "Every vector is a combination of these basis vectors."
        ]
        self.setup_layout(title_text, lecture_lines)
        
        # Color definitions
        COLOR_I = "#00FF00"  # Green
        COLOR_J = "#FF0000"  # Red
        COLOR_GRID = "#444444"
        COLOR_VECTOR = "#FFFFFF" # White

        # Origin point at D2 (standard origin for this section)
        origin_coords = self.grid["D2"]

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(COLOR_I), run_time=0.5)
        
        # Grid range chosen to fit A1-F6 when origin is at D2
        grid = NumberPlane(
            x_range=[-1, 4, 1],
            y_range=[-2, 3, 1],
            background_line_style={
                "stroke_color": COLOR_GRID,
                "stroke_width": 2,
                "stroke_opacity": 0.6
            },
            axis_config={"include_tip": False, "stroke_color": COLOR_GRID}
        )
        # Using place_in_area to center the grid correctly
        self.place_in_area(grid, "A1", "F6")

        # Basis vectors relative to origin (D2)
        # i_vec ends at D3 (1 unit right)
        i_vec = Arrow(
            start=origin_coords,
            end=self.grid["D3"],
            buff=0,
            color=COLOR_I,
            stroke_width=4
        )
        # j_vec ends at C2 (1 unit up)
        j_vec = Arrow(
            start=origin_coords,
            end=self.grid["C2"],
            buff=0,
            color=COLOR_J,
            stroke_width=4
        )
        
        # Labels using place_at_grid as requested by Issues 26 and 27
        label_i = Text("i", slant=ITALIC, color=COLOR_I, font_size=24)
        self.place_at_grid(label_i, 'E3', scale_factor=0.8)
        
        label_j = Text("j", slant=ITALIC, color=COLOR_J, font_size=24)
        self.place_at_grid(label_j, 'C1', scale_factor=0.8)

        self.play(Create(grid), run_time=1.5)
        self.play(
            GrowArrow(i_vec), 
            GrowArrow(j_vec), 
            Write(label_i), 
            Write(label_j),
            run_time=1
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(COLOR_J), run_time=0.5)
        
        # Flash the i and j vectors to highlight them as basis
        for _ in range(2):
            self.play(
                Flash(i_vec, color=COLOR_I, line_length=0.3, flash_radius=0.5),
                Flash(j_vec, color=COLOR_J, line_length=0.3, flash_radius=0.5),
                run_time=0.5
            )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(COLOR_VECTOR), run_time=0.5)
        
        # Vector v = 3i + 2j ends at B5 (3 units right, 2 units up from D2)
        target_coords = self.grid["B5"]
        
        main_vector = Arrow(
            start=origin_coords,
            end=target_coords,
            buff=0,
            color=COLOR_VECTOR,
            stroke_width=6
        )
        
        # Vector formula using place_in_area as requested by Issue 28
        v_formula = Text("v = 3i + 2j", slant=ITALIC, color=COLOR_VECTOR, font_size=24)
        self.place_in_area(v_formula, 'A5', 'B6', scale_factor=0.9)
        
        # Component lines to show projection (3i and 2j)
        # Horizontal component: D2 to D5
        i_comp_line = DashedLine(
            start=origin_coords,
            end=self.grid["D5"],
            color=COLOR_I,
            stroke_width=2
        ).set_opacity(0.7)
        
        # Vertical component: D5 to B5
        j_comp_line = DashedLine(
            start=self.grid["D5"],
            end=target_coords,
            color=COLOR_J,
            stroke_width=2
        ).set_opacity(0.7)

        self.play(Create(i_comp_line), run_time=0.8)
        self.play(Create(j_comp_line), run_time=0.8)
        self.play(
            GrowArrow(main_vector), 
            Write(v_formula), 
            run_time=1.2
        )
        self.wait(2)
