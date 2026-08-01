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
        # Setup the title and lecture lines
        title_str = "Prerequisite Setup: The 2D Foundation"
        lines_str = [
            "Meet Vector-Bot, navigating via two 2D vectors, u and v.",
            "These vectors form a parallelogram on the ground.",
            "The 2D determinant measures this area's signed scaling factor."
        ]
        self.setup_layout(title_str, lines_str)

        # Colors
        u_color = "#52CEFF"
        v_color = "#C6FF00"
        area_color = "#FFD700"

        # === Animation for Lecture Line 1 ===
        # 1. 2D Coordinate system
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            x_length=4.5,
            y_length=4.5,
            axis_config={"include_tip": True, "color": GREY_B},
            tips=False
        )
        # Position axes in the middle of the right-side grid
        self.place_in_area(axes, "A1", "F6", scale_factor=1.0)
        
        # 2. Vector-Bot representation at origin
        origin_pos = axes.c2p(0,0)
        drone_body = Dot(origin_pos, color=WHITE, radius=0.08)
        propellers = VGroup(*[
            Dot(origin_pos, color=GREY_A, radius=0.03).shift(0.1 * (RIGHT*np.cos(a) + UP*np.sin(a)))
            for a in [PI/4, 3*PI/4, 5*PI/4, 7*PI/4]
        ])
        drone = VGroup(drone_body, propellers)

        # 3. Vectors u and v
        u_coords = [2, 0.5, 0]
        v_coords = [0.5, 2, 0]
        
        vec_u = Arrow(axes.c2p(0,0), axes.c2p(*u_coords), buff=0, color=u_color)
        vec_v = Arrow(axes.c2p(0,0), axes.c2p(*v_coords), buff=0, color=v_color)
        
        # Use Text instead of MathTex to avoid LaTeX dependency error
        label_u = Text("u", color=u_color, slant=ITALIC)
        label_v = Text("v", color=v_color, slant=ITALIC)
        
        # Position labels using grid system
        self.place_at_grid(label_u, "C5", scale_factor=0.6)
        self.place_at_grid(label_v, "A4", scale_factor=0.6)

        # Execution
        self.lecture[0].set_color(u_color)
        self.play(Create(axes), FadeIn(drone))
        self.play(GrowArrow(vec_u), Write(label_u))
        self.play(GrowArrow(vec_v), Write(label_v))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(area_color)
        
        # Get scene coordinates for parallelogram
        p0 = axes.c2p(0,0)
        p1 = axes.c2p(*u_coords)
        p2 = axes.c2p(*(np.array(u_coords) + np.array(v_coords)))
        p3 = axes.c2p(*v_coords)
        
        parallelogram = Polygon(
            p0, p1, p2, p3, 
            fill_opacity=0.3, 
            fill_color=area_color, 
            stroke_width=2, 
            stroke_color=area_color
        )
        
        # Guide lines
        ghost_u = DashedLine(p3, p2, color=u_color)
        ghost_v = DashedLine(p1, p2, color=v_color)
        
        self.play(Create(ghost_u), Create(ghost_v))
        self.play(FadeIn(parallelogram))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Area label (Using Text instead of MathTex)
        det_label = Text("Area = det(u, v)", color=area_color)
        # Position label in a cell area to avoid overlap with parallelogram
        self.place_in_area(det_label, 'A1', 'A3', scale_factor=0.6)
        
        self.play(Write(det_label))
        self.wait(2)
