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
        # Setup layout
        title = "Visualizing High Dimensions & Subspaces"
        lines = [
            "Vector spaces can have infinite dimensions, like color palettes.",
            "RGB values act as basis vectors for any color.",
            "Subspaces are smaller planes living within larger spaces."
        ]
        self.setup_layout(title, lines)

        # === Animation for Lecture Line 1 ===
        # Show three spheres: Red (#FF0000), Green (#00FF00), and Blue (#0000FF) labeled 'Basis'
        # alongside a palette icon [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/palette.svg].
        self.lecture[0].set_color(YELLOW)
        
        palette_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/palette.svg")
        self.place_at_grid(palette_icon, "B2", scale_factor=0.6)
        
        red_sphere = Dot(color="#FF0000", radius=0.4)
        green_sphere = Dot(color="#00FF00", radius=0.4)
        blue_sphere = Dot(color="#0000FF", radius=0.4)
        
        # Grid positions from Issue 33
        self.place_at_grid(red_sphere, "B3")
        self.place_at_grid(green_sphere, "B4")
        self.place_at_grid(blue_sphere, "B5")
        
        basis_label = Text("Basis", font_size=20, color=WHITE)
        self.place_at_grid(basis_label, "A4")
        
        spheres = VGroup(red_sphere, green_sphere, blue_sphere)
        self.play(FadeIn(palette_icon), FadeIn(spheres), Write(basis_label))
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        # Blend them to create a new color point in a 3D coordinate system.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Create a simplified 3D coordinate system (Axes)
        axes = Axes(
            x_range=[0, 5, 1],
            y_range=[0, 5, 1],
            x_length=3,
            y_length=3,
            axis_config={"color": BLUE_E},
            tips=True
        )
        # Manual z-axis for 3D appearance
        z_axis = Line(axes.c2p(0,0,0), axes.c2p(0,0,0) + np.array([-1, -1, 0]), color=BLUE_E).add_tip()
        coord_sys = VGroup(axes, z_axis)
        
        # Grid area from Issue 32
        self.place_in_area(coord_sys, "C3", "F6", scale_factor=0.7)
        
        # Target point: a blended color (Orange)
        target_color = "#FFA500"
        blended_point = Dot(color=target_color, radius=0.15)
        blended_point.move_to(axes.c2p(3, 2, 0))
        
        point_label = Text("(R, G, B)", font_size=16, color=WHITE)
        point_label.next_to(blended_point, UR, buff=0.1)

        self.play(
            FadeOut(basis_label),
            FadeOut(palette_icon),
            spheres.animate.scale(0.2).move_to(axes.c2p(0,0,0)),
            FadeIn(coord_sys)
        )
        self.play(
            ReplacementTransform(spheres, blended_point),
            Write(point_label)
        )
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # Draw a 3D cube and slice it with a translucent 2D plane (#FFFFFF, 0.3 opacity) labeled 'Subspace'.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Create a wireframe cube
        origin_3d = axes.c2p(2,2,0)
        offset = np.array([-0.5, -0.5, 0])
        cube_front = Square(side_length=1.5, color=BLUE_B).move_to(origin_3d)
        cube_back = Square(side_length=1.5, color=BLUE_B).move_to(origin_3d + offset)
        cube_edges = VGroup(
            Line(cube_front.get_corner(UL), cube_back.get_corner(UL), color=BLUE_B),
            Line(cube_front.get_corner(UR), cube_back.get_corner(UR), color=BLUE_B),
            Line(cube_front.get_corner(DL), cube_back.get_corner(DL), color=BLUE_B),
            Line(cube_front.get_corner(DR), cube_back.get_corner(DR), color=BLUE_B)
        )
        cube = VGroup(cube_front, cube_back, cube_edges)
        
        # Subspace plane
        subspace_plane = Polygon(
            axes.c2p(0, 0, 0),
            axes.c2p(4, 1, 0),
            axes.c2p(5, 4, 0),
            axes.c2p(1, 3, 0),
            fill_color=WHITE,
            fill_opacity=0.3,
            stroke_width=1,
            color=WHITE
        )
        
        subspace_text = Text("Subspace", font_size=20, color=WHITE)
        # Grid position from Issue 31
        self.place_at_grid(subspace_text, "B6")

        self.play(
            FadeOut(blended_point),
            FadeOut(point_label),
            Create(cube)
        )
        self.play(
            FadeIn(subspace_plane),
            Write(subspace_text)
        )
        self.wait(3)

        # Reset colors
        self.lecture[2].set_color(WHITE)
        self.wait(1)
